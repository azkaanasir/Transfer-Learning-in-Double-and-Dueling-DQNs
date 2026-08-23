"""Q-network builders, dueling aggregation variants, and named layer groups.

Both architectures share an identically-named trunk (`trunk_fc1`, `trunk_fc2`)
so that weight transfer keys on layer *names*. The original implementation
zipped source and target Dense layers positionally, which silently dropped
layers whenever the two architectures had different depths -- see
paper/METHODS_ACTUAL.md section 5.1.

    mlp      state -> trunk_fc1(128) -> trunk_fc2(128) -> q_out(|A|)

    dueling  state -> trunk_fc1(128) -> trunk_fc2(128) -> value_fc(64) -> value_out(1)
                                                       \\-> adv_fc(64)  -> adv_out(|A|)

`LAYER_GROUPS` is the other half of that fix. The published freeze logic
resolved its target set by *indexing into* `model.layers`, and for a branched
functional model that order put `value_fc` and `adv_fc` adjacent -- so a rule
meant to keep the trunk trainable froze both head hiddens instead, and nobody
could tell from reading the config. Every freeze and transfer set in this
codebase is named through `LAYER_GROUPS` or through explicit layer names, and
`resolve_layers` fails loudly on anything it does not recognise.
"""
from __future__ import annotations

from typing import Any, Iterable, Sequence

import tensorflow as tf
import keras
from keras import layers

TRUNK_LAYERS = ('trunk_fc1', 'trunk_fc2')
AGGREGATIONS = ('mean', 'max', 'naive')

# Semantic names for sets of layers, per architecture. An experiment definition
# names a group; nothing anywhere resolves a layer by position.
LAYER_GROUPS: dict[str, dict[str, tuple[str, ...]]] = {
    'mlp': {
        'trunk': ('trunk_fc1', 'trunk_fc2'),
        'trunk_fc1': ('trunk_fc1',),
        'trunk_fc2': ('trunk_fc2',),
        'head': ('q_out',),
        'all': ('trunk_fc1', 'trunk_fc2', 'q_out'),
        'none': (),
    },
    'dueling': {
        'trunk': ('trunk_fc1', 'trunk_fc2'),
        'trunk_fc1': ('trunk_fc1',),
        'trunk_fc2': ('trunk_fc2',),
        # the two streams, split so a stream-wise ablation can name exactly one
        'value': ('value_fc', 'value_out'),
        'value_hidden': ('value_fc',),
        'value_out': ('value_out',),
        'adv': ('adv_fc', 'adv_out'),
        'adv_hidden': ('adv_fc',),
        'adv_out': ('adv_out',),
        'head_hiddens': ('value_fc', 'adv_fc'),
        'heads': ('value_fc', 'value_out', 'adv_fc', 'adv_out'),
        'all': ('trunk_fc1', 'trunk_fc2', 'value_fc', 'value_out',
                'adv_fc', 'adv_out'),
        'none': (),
    },
}


def layer_names(arch: str) -> tuple[str, ...]:
    """Every weighted layer of an architecture, in forward order."""
    return LAYER_GROUPS[arch]['all']


def resolve_layers(arch: str, names: Iterable[str]) -> tuple[str, ...]:
    """Expand group names and bare layer names into concrete layer names.

    Order is the architecture's forward order, and duplicates collapse, so that
    ('trunk', 'trunk_fc1') and ('trunk_fc2', 'trunk_fc1') both resolve to a
    canonical tuple. That canonical form is what run identity and the manifest
    record, which is what makes two nominally-different ablation specs
    detectably identical rather than quietly producing duplicate arms.
    """
    if arch not in LAYER_GROUPS:
        raise ValueError(f'unknown arch {arch!r}')
    groups = LAYER_GROUPS[arch]
    forward = groups['all']
    chosen: set[str] = set()
    for name in names:
        key = str(name)
        if key in groups:
            chosen.update(groups[key])
        elif key in forward:
            chosen.add(key)
        else:
            raise ValueError(
                f'{arch}: {key!r} is neither a layer nor a layer group. '
                f'Layers: {forward}. Groups: {sorted(groups)}. '
                f'Refusing to guess -- a mis-resolved layer set is exactly the '
                f'defect that invalidated the published freeze schedule.')
    return tuple(n for n in forward if n in chosen)


@keras.saving.register_keras_serializable(package='dqn')
class DuelingAggregation(layers.Layer):
    """Combine the value and advantage streams into Q.

    mode='mean'   Q = V + (A - mean_a' A)      the Wang et al. (2016) form
    mode='max'    Q = V + (A - max_a' A)       the identifiable form, also in that paper
    mode='naive'  Q = V + A                    unidentifiable: V and A are only
                                               determined up to a constant, so
                                               this is the arm that tests whether
                                               the baseline subtraction is doing
                                               the work attributed to it

    The mode is part of the serialised config, so a checkpoint cannot be
    reloaded under a different aggregation without the mismatch being visible.
    """

    def __init__(self, mode: str = 'mean', **kwargs):
        super().__init__(**kwargs)
        if mode not in AGGREGATIONS:
            raise ValueError(f'aggregation must be one of {AGGREGATIONS}, '
                             f'got {mode!r}')
        self.mode = mode

    def call(self, inputs):
        value, advantage = inputs
        if self.mode == 'mean':
            baseline = tf.reduce_mean(advantage, axis=1, keepdims=True)
        elif self.mode == 'max':
            baseline = tf.reduce_max(advantage, axis=1, keepdims=True)
        else:
            return value + advantage
        return value + (advantage - baseline)

    def get_config(self):
        cfg = dict(super().get_config())
        cfg['mode'] = self.mode
        return cfg


def build_q_network(state_dim: int,
                    action_dim: int,
                    arch: str = 'dueling',
                    hidden: Sequence[int] = (128, 128),
                    head_units: int = 64,
                    aggregation: str = 'mean',
                    layer_seeds: dict[str, int] | None = None) -> keras.Model:
    """Build an uncompiled Q-network.

    Training runs through an explicit `GradientTape`, so the model is
    deliberately left uncompiled: that is what lets `layer.trainable` be toggled
    mid-run without discarding the optimiser state, which the manuscript's
    freeze-then-unfreeze protocol requires and the published code never did.

    `layer_seeds` maps layer name to initialiser seed (see `seeding.Seeds`).
    Seeding per layer rather than globally has a deliberate consequence: at a
    given run seed an `mlp` and a `dueling` network receive *identical* trunk
    initialisations, so the architecture contrast carries no trunk-init noise
    and the design is matched by seed. Passing None falls back to the framework
    default, which is reproducible only through global state and is therefore
    for throwaway probes, not for recorded runs.
    """
    if len(hidden) != 2:
        raise ValueError(f'expected two trunk widths, got {hidden!r}')

    def init(name):
        if not layer_seeds:
            return 'glorot_uniform'
        if name not in layer_seeds:
            raise KeyError(
                f'no initialiser seed supplied for layer {name!r}. Partial '
                f'seeding is worse than none, because it looks reproducible '
                f'and is not.')
        return keras.initializers.GlorotUniform(seed=int(layer_seeds[name]))

    inputs = layers.Input(shape=(state_dim,), name='state')
    x = layers.Dense(hidden[0], activation='relu', name='trunk_fc1',
                     kernel_initializer=init('trunk_fc1'))(inputs)
    x = layers.Dense(hidden[1], activation='relu', name='trunk_fc2',
                     kernel_initializer=init('trunk_fc2'))(x)

    if arch == 'mlp':
        outputs = layers.Dense(action_dim, name='q_out',
                               kernel_initializer=init('q_out'))(x)
    elif arch == 'dueling':
        v = layers.Dense(head_units, activation='relu', name='value_fc',
                         kernel_initializer=init('value_fc'))(x)
        v = layers.Dense(1, name='value_out',
                         kernel_initializer=init('value_out'))(v)
        a = layers.Dense(head_units, activation='relu', name='adv_fc',
                         kernel_initializer=init('adv_fc'))(x)
        a = layers.Dense(action_dim, name='adv_out',
                         kernel_initializer=init('adv_out'))(a)
        outputs = DuelingAggregation(mode=aggregation,
                                     name='dueling_aggregation')([v, a])
    else:
        raise ValueError(f'unknown arch {arch!r}')

    return keras.Model(inputs, outputs, name=f'{arch}_q')


TRANSFER_SETS = ('matched', 'described', 'trunk', 'fc1', 'fc2', 'none')


def shape_exact_layers(arch: str, src_obs: int, tgt_obs: int,
                       src_act: int, tgt_act: int) -> tuple[str, ...]:
    """Layers whose kernel shape is *identical* in the source and target nets.

    `trunk_fc1` depends on the observation dimension; `q_out` and `adv_out` on
    the action cardinality. Everything else is interface-independent. Under an
    interface change this set therefore excludes the input-facing layer as well
    as the output head, so nothing is partial-copied.
    """
    out = []
    for name in LAYER_GROUPS[arch]['all']:
        if name == 'trunk_fc1':
            ok = src_obs == tgt_obs
        elif name in ('q_out', 'adv_out'):
            ok = src_act == tgt_act
        else:
            ok = True
        if ok:
            out.append(name)
    return tuple(out)


def output_mismatched_layers(arch: str, src_act: int,
                             tgt_act: int) -> tuple[str, ...]:
    """Layers whose *output* width is forced to change by the action cardinality."""
    if src_act == tgt_act:
        return ()
    return tuple(n for n in LAYER_GROUPS[arch]['all']
                 if n in ('q_out', 'adv_out'))


def transfer_set_layers(arch: str, transfer_set: str,
                        src_obs: int, tgt_obs: int,
                        src_act: int, tgt_act: int) -> tuple[str, ...]:
    """Resolve a named transfer set into concrete layer names.

    Named sets rather than raw layer lists, because a raw list is not
    comparable across architectures. {trunk_fc1, trunk_fc2} copies 94 per cent
    of the mlp's parameters and 50 per cent of the dueling network's, which
    confounds the architecture factor with treatment intensity. Adversarial
    review found that defect in revision 1 of the design; it is the same class
    of error the Phase 0 audit found in the published study, and it is why the
    primary level is defined by *matched intensity* rather than by layer names.

    The levels, and what each one means scientifically:

    matched    Everything except the layers whose output width the action
               cardinality forces to change. The input-facing layer is
               partial-copied. Transferred fraction is then comparable across
               architectures (0.94 vs 0.98 for CartPole -> LunarLander), which
               is what makes the `arch` contrast identified. **The primary
               level.**

    described  Only layers whose shapes match exactly, so the input *and*
               output layers are reinitialised and nothing is partial-copied.
               This is the protocol the manuscript *describes* -- "the input and
               output layers were reinitialized to match LunarLander's
               8-dimensional state space and 4-action output".

    trunk      The trunk only, with the input-facing layer partial-copied. This
               is the protocol the published code actually *implemented*.
               Retained as a pre-declared secondary, and the `described` versus
               `trunk` contrast is the measurable form of the code-versus-
               manuscript discrepancy the Phase 0 audit documented.

    fc1, fc2   Single-layer ablation levels.
    none       Nothing transferred; a consistency check against scratch.
    """
    if transfer_set == 'matched':
        forced = set(output_mismatched_layers(arch, src_act, tgt_act))
        return tuple(n for n in LAYER_GROUPS[arch]['all'] if n not in forced)
    if transfer_set == 'described':
        return shape_exact_layers(arch, src_obs, tgt_obs, src_act, tgt_act)
    if transfer_set == 'trunk':
        return resolve_layers(arch, ['trunk'])
    if transfer_set == 'fc1':
        return resolve_layers(arch, ['trunk_fc1'])
    if transfer_set == 'fc2':
        return resolve_layers(arch, ['trunk_fc2'])
    if transfer_set == 'none':
        return ()
    raise ValueError(f'unknown transfer_set {transfer_set!r}; expected one of '
                     f'{TRANSFER_SETS}')


def build_stream_probe(model: keras.Model) -> keras.Model | None:
    """A model exposing (V, A) for diagnostics. None for non-dueling nets.

    Mechanism claims in `DESIGN.md` section 5.5 must cite an instrumented
    signal, and V/A magnitudes are the signal for the value-scale hypothesis.
    Capturing them needs the intermediate tensors, which is cheap to wire up at
    construction and awkward afterwards.
    """
    try:
        v = model.get_layer('value_out').output
        a = model.get_layer('adv_out').output
    except ValueError:
        return None
    return keras.Model(model.input, [v, a], name='stream_probe')


def build_feature_probe(model: keras.Model,
                        layer: str = 'trunk_fc2') -> keras.Model:
    """A model exposing one hidden layer's activations.

    The input to the representation measurements: CKA between a transferred
    trunk and the matched-seed scratch trunk on one fixed batch of target
    states, CKA drift of the trunk over training, and the inactive-unit
    fraction a frozen transferred layer can drift into.
    """
    return keras.Model(model.input, model.get_layer(layer).output,
                       name=f'feature_probe_{layer}')


CUSTOM_OBJECTS: dict[str, Any] = {
    'DuelingAggregation': DuelingAggregation,
    # the published checkpoints used this name; kept so those models still load
    'DuelingCombineLayer': DuelingAggregation,
}


__all__ = ['build_q_network', 'build_stream_probe', 'build_feature_probe',
           'DuelingAggregation', 'CUSTOM_OBJECTS', 'TRUNK_LAYERS',
           'AGGREGATIONS', 'LAYER_GROUPS', 'layer_names', 'resolve_layers',
           'shape_exact_layers', 'output_mismatched_layers',
           'transfer_set_layers',
           'TRANSFER_SETS']
