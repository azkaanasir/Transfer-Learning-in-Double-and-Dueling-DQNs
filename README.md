# RL- Final Project  
Transfer Learning in Deep Q-Networks: A Controlled Study of DDQN and Dueling DQN Under Cross-Environment Transfer

#  Transfer Learning in Deep Q-Networks

This repository contains the final report for our project titled:

**A Controlled Study of Double DQN and Dueling DQN Under Cross-Environment Transfer**

📍 *Authors: Azkaa Nasir, Fatima Dossa, Muhammad Ahmed Atif*  
🏫 *Dhanani School of Science and Engineering, Habib University*

---

##  Project Summary

In this study, we conducted a **controlled empirical comparison** of two value-based deep reinforcement learning architectures — **Double DQN (DDQN)** and **Dueling DQN** — under both single-task and cross-environment transfer settings.

We trained both models on **CartPole-v2** (source task), then transferred learned representations to **LunarLander-v3** (target task) using a fixed layer-wise transfer protocol.

Our primary objective was to isolate how **architectural inductive bias** influences transfer robustness under substantial domain shift.

### Focus Areas

- Cross-environment transfer behavior  
- Stability under domain shift  
- Validation reward comparison  
- Statistical significance across random seeds  

---

## Contents

- `RL_Project_Report.pdf` – Complete research paper including experimental design, statistical analysis, and results

---

## Highlights

-  **DDQN demonstrated robust transfer behavior**, maintaining stable learning and achieving positive rewards comparable to training from scratch.
-  **Dueling DQN exhibited severe negative transfer**, with degraded rewards and unstable optimization.
-  Statistical testing (Mann–Whitney U) confirmed a **significant performance gap (p < 0.01)** between architectures under transfer.
-  Architectural bias reduction (DDQN) proved more transfer-robust than advantage-value decomposition (Dueling DQN).

---

## Key Findings

- Transfer effectiveness is **architecture-dependent**.
- Improvements optimized for single-task learning do not necessarily generalize under domain shift.
- Bias-reduction mechanisms appear more stable than structural value decomposition when reusing representations.

---

Feel free to read the full report for detailed methodology, experimental setup, statistical analysis, limitations, and future research directions.
