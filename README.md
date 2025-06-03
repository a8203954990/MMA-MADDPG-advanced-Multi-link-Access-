# MMA-MADDPG-advanced-Multi-link-Access
[Throughput-Optimal Multi-Link Access for Wi-Fi 7 via Multi-Agent Reinforcement Learning](https://ieeexplore.ieee.org/document/10978161)

Our paper has been accepted by 2025 IEEE Wireless Communications and Networking Conference (WCNC)

Thanks for your interest in our paper. This project implements a Multi-Agent Deep Deterministic Policy Gradient (MADDPG) solution to optimize throughput in multi-link Wi-Fi networks. The system learns access strategies that maximize channel utilization while minimizing collisions.

Repository Structure

├── model/              # Trained Actor network models

│   ├── [.pt files]     # PyTorch models (state → action mapping)

│   └── [.onnx files]   # Optional ONNX-format models

│

├── outs/               # Performance visualization

│   └── throughput_plots/  # Node-wise and total throughput graphs

│

├── rewards/            # Training reward logs

│   └── [.txt files]    # Raw transmission success records per timeslot

│

├── src/                # Source code

│   ├── rl_utils.py        # Reinforcement learning utilities

│   ├── env_singlelink.py  # Wireless channel simulation environment

│   ├── MADDPG_SingleLink.py  # Training script (single-link)

│   ├── evaluate.py        # Model evaluation script

│   └── plot_reward.py     # Throughput visualization script

│

└── README.md           # This document

Key Components
Core Scripts

env_singlelink.py:


Virtual Environment in Reinforcement Learning
A simulated channel environment that abstracts real-world wireless communication conditions. It models channel behaviors including packet collisions and successful transmissions.

MADDPG_SingleLink.py:

Training Script
Implements the end-to-end MADDPG training pipeline exclusively in a single-link environment. Generates:
Training reward logs (rewards/ directory)
Optimized Actor network model (PyTorch .pt format in model/ directory)
Training process:
Learns collision-avoidance transmission policies
Maximizes channel utilization through RL
Outputs portable network models for deployment

evaluate.py:

Loads a pre-trained Actor network model (single-link environment) and interacts with the channel simulator (env_singlelink.py). Executes performance testing under a 3-link configuration and outputs throughput metrics.

plot_reward.py:

Visualization:
Converts rewards/ .txt logs to throughput plots
Outputs graphs to outs/

Supporting File
rl_utils.py
Essential DRDL helper functions (no modification required)

Dependencies
Package	Version
Python	3.x
PyTorch	1.12.1
torchvision	0.13.1
numpy	1.23.3
tqdm	4.61.1
matplotlib	3.6.0

Usage Workflow
1. Training (Optional)
bash
python src/MADDPG_SingleLink.py
Outputs:

Trained model (model/*.pt)

Reward logs (rewards/*.txt)

2. Evaluation
bash
python src/evaluate.py
Expected result:
Throughput approaching saturation in 3-link configuration
Throughput Example

3. Visualization
bash
python src/plot_reward.py
Generates throughput curves from reward logs

Performance Notes
Ideal evaluation output shows near-saturation total throughput

If pretrained models underperform:

Retrain with MADDPG_SingleLink.py

Save renamed model to model/

Verify format compatibility for evaluation

Run evaluate.py to validate improvements
