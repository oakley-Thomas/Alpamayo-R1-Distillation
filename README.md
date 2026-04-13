# Alpamayo-R1 Distillation Analysis

Building on the work from [Muhammad Hashmi](https://github.com/mu-hashmi)

- [Blog Post](https://substack.com/home/post/p-186795679)
- [Implementation](https://github.com/mu-hashmi/alpamayo-r1-distilled)

---

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if you don't have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

Install dependencies:

```bash
uv sync
```

---

## Dataset

This project uses chunk 0 of the [NVIDIA Physical AI Open Dataset](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles) (~100 driving clips), downloading only the 4 camera feeds required by Alpamayo-R1 (front-wide, front-tele, cross-left, cross-right) plus egomotion labels and calibration.

### Prerequisites

1. Accept the dataset license at [huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles)
2. Log in with your HuggingFace token:

```bash
uv run huggingface-cli login
```

### Download

```bash
uv run python download_data.py
```

Data will be extracted to `data/`. This folder is excluded from version control — re-run the script to repopulate it.
