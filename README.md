# AMAS Assignment 2

## Overview

This repository contains:
- Part 1: Independent Q-Learning (IQL) and Centralized Q-Learning (CQL) on PettingZoo Pursuit.
- Part 2: Custom Level-Based Foraging environment with global and mixed reward functions.

## Files

- `run_all.py`: Main experiment runner for Part 1 and Part 2.
- `requirements.txt`: Python dependencies.
- `report.pdf`: Assignment report summary.
- `figures/`: Generated plot outputs.
- `results/`: Pickled quick experiment results.
- `amas_a2/part1/`: Part 1 source code.
- `amas_a2/part2/`: Part 2 source code.

## Reproduction

1. Activate the local environment:

```bash
cd /home/AMAS/amas_a2
. .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run a quick experiment:

```bash
PYTHONPATH=/home/AMAS/amas_a2 python run_all.py --quick
```

4. Run a faster ~1 hour demo experiment on GPU (3 seeds, limited episodes):

```bash
PYTHONPATH=/home/AMAS/amas_a2 python run_all.py --fast --device cuda
```

5. View outputs:

- `figures/part1_iql_quick.png`
- `figures/part1_cql_quick.png`
- `figures/part2_global_quick.png`
- `figures/part2_mixed_quick.png`
- `results/*.pkl`
- `report.pdf`
