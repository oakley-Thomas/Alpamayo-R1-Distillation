# Alpamayo-R1 Action Prediction Distillation — Project Summary

## Overview

This project explores **knowledge distillation** of NVIDIA's **Alpamayo-R1**, a 10-billion parameter Vision-Language-Action (VLA) model for autonomous driving. The goal is to train a smaller "student" model that mimics the trajectory prediction behavior of the larger "teacher" model, then benchmark both to understand the performance trade-offs at lower compute cost.
---

## Background

**Alpamayo-R1** is NVIDIA's open-source VLA model that uses chain-of-thought reasoning to predict driving trajectories from camera input. It is built on a Cosmos Reason backbone (8.2B parameters) plus a 2.3B action expert, requiring at least 24 GB VRAM to run.

**AlpaSim** is NVIDIA's companion open-source simulation framework for evaluating AV policies in a closed-loop reactive environment.

**Knowledge Distillation** is a model compression technique where a smaller "student" model is trained to mimic the outputs of a larger "teacher" model, rather than learning from raw ground truth labels alone.

---

## Project Scope

- **Scope:** Action prediction distillation only (chain-of-thought reasoning distillation is out of scope for this prototype)
- **Dataset:** Subset of the [NVIDIA Physical AI Open Dataset](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles) (~100 driving clips)
- **Teacher:** Alpamayo-R1 (10B parameters)
- **Student:** Smaller transformer model (target: 2B–4B parameters)
- **Evaluation:** Open-loop benchmark — compare teacher vs. student trajectory predictions against ground truth using **minADE** and **RMSE**

---

## 4-Week Plan

### Week 1 — Setup & Baseline Profiling
- Clone [Alpamayo GitHub repo](https://github.com/NVlabs/alpamayo) and load model from HuggingFace
- Run teacher inference on ~100 clips from the physical AI dataset
- Profile teacher: inference latency, VRAM usage, trajectory prediction quality
- Establish baseline metrics (minADE, RMSE)

### Week 2 — Student Architecture & Distillation Pipeline
- Select student model architecture (2B–4B parameter transformer)
- Implement distillation training loop:
  - KL divergence loss on teacher trajectory logits (soft targets)
  - Cross-entropy loss on ground truth trajectories (hard targets)
- Set up data loading and training infrastructure
- Launch first training run

### Week 3 — Training & Iteration
- Iteratively tune distillation temperature, learning rate, batch size
- Train student model to convergence on dataset subset
- Evaluate intermediate checkpoints against teacher performance
- Identify sweet spot between model size and trajectory accuracy

### Week 4 — Evaluation & Writeup
- Benchmark teacher vs. student on held-out test clips
- Measure: trajectory error (minADE, RMSE), inference speed, memory footprint
- Produce comparison table and analysis
- Write up findings: what performance is lost at each compression ratio?

---

## Key References

- [*"Distilling Alpamayo-R1: What I learned making a 10-Billion parameter VLA model 20x smaller"*](https://substack.com/home/post/p-186795679)
- [Alpamayo GitHub](https://github.com/NVlabs/alpamayo)
- [AlpaSim GitHub](https://github.com/NVlabs/alpasim)
- [Alpamayo-R1 on HuggingFace](https://huggingface.co/nvidia/Alpamayo-R1-10B)
- [NVIDIA Physical AI Dataset](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles)
- [NVIDIA Alpamayo Developer Page](https://developer.nvidia.com/drive/alpamayo)

---

## Stretch Goal (Time Permitting)

If action prediction distillation is completed ahead of schedule, explore **reasoning distillation** — training the student to also reproduce Alpamayo's chain-of-thought reasoning traces alongside trajectory predictions. This is a novel contribution not yet addressed in the existing literature.

---

*Project by Oakley Thomas | CSE676 — University at Buffalo | Spring 2026*
