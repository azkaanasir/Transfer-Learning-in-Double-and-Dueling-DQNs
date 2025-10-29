"""Entrypoint for running training scripts."""

import argparse
import os

from config import TB_LOG_DIR, PRETRAINED_CARTPOLE_PATH
from train import train_agent as train_lunar

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', choices=['lunar'], default='lunar', help='Which environment to train')
    parser.add_argument('--no-tb', action='store_true', help='Disable TensorBoard log directory creation')
    parser.add_argument('--pretrained', type=str, default=PRETRAINED_CARTPOLE_PATH, help='Path to pretrained cartpole model (.h5) to use for transfer (optional)')
    parser.add_argument('--no-freeze', action='store_true', help='Do NOT freeze base trunk (useful if you want to fine-tune immediately)')
    args = parser.parse_args()

    if not args.no_tb:
        os.makedirs(TB_LOG_DIR, exist_ok=True)

    if args.env == 'lunar':
        train_lunar(pretrained_path=args.pretrained, freeze_base=not args.no_freeze)
    else:
        raise SystemExit('Unsupported env')

if __name__ == '__main__':
    main()
