# Surrogate-Assisted Deep Reinforcement Learning for Gear Center Distance Optimization

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Official implementation of the paper:  
**"A Surrogate-Assisted Deep Reinforcement Learning Framework for Gear Center Distance Optimization in Machine Tool Assembly"**.

---

## 📖 Overview

Precision adjustment of gear center distance during machine tool assembly is traditionally labor-intensive, requiring repeated on-rig measurements (often 41 discrete shim adjustments). This repository provides a **Surrogate-Assisted Deep Reinforcement Learning (DRL)** framework that:
1. **Surrogate Model (Data-Driven Virtual Environment):** Employs an optimized **Random Forest** regression model to simulate gear deviation responses across 8 standard metrics compliant with **ANSI/AGMA 2015-1-A01**.
2. **Residual Dueling DQN Agent:** Explores optimal micro-adjustments using:
   - **Residual Connections & LayerNorm** for deep feature extraction.
   - **Visited Mask** mechanism to eliminate cyclic exploration loops.
   - **Customized Dense Reward Shaping** to guide error reduction gradients.
   - **Expert Demonstration** to mitigate cold-start issues.
   - **Prioritized Experience Replay (PER)** and **Dynamic $\epsilon$-Greedy Scheduling**.

Experimental results show an optimization success rate of **92.0%** and reduce the physical on-rig measurement requirements from **41 to 1** (a 41:1 efficiency improvement).

---

## 📁 Repository Structure

```text
├── Complete_model/                                 # Trained DRL model checkpoints and saved replay buffer files
├── data_set_RF_DD_test/                            # Gear test profiles and datasets for policy evaluation (CSV)
├── data_set_RF_DD_train/                           # Industrial gear measurement datasets for DRL training (CSV)
├── surrogate_model/                                # Surrogate environment module and workspace
│   ├── processed_data/                             # Preprocessed gear measurement transition samples for training (CSV)
│   └── surrogate_comparison_optimization.py        # Surrogate model benchmarking and hyperparameter optimization
├── DQN_doubleDQN_duelingDQN_ablate_deep_Residual.py # Main script for DRL policy training, ablation, and inference
├── requirements.txt                                # Python package dependencies
└── README.md                                       # Project documentation
