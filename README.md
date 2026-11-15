# Multi-agent Q-learning in pursuit and foraging

Two experiments examine how agents learn to cooperate. In PettingZoo Pursuit, the comparison is between independent Q-learning and centralized Q-learning. In a custom level-based foraging environment, the comparison is between a shared global reward and a mixed reward. The first changes how decisions are learned; the second changes the feedback available to each learner.

## Reproduce the experiments

From the repository root, activate a Python environment and run:

```sh
python -m pip install -r requirements.txt
python run_all.py --quick --device cpu
```

The quick run uses fewer episodes. For the longer preset, use `python run_all.py --fast --device cpu`, or select `--device cuda` with a CUDA-enabled PyTorch installation. The runner writes learning curves to `figures/` and serialized experiment results to `results/`.

## Read the implementation

- [Pursuit agents](amas_a2/part1/agent.py) and [training loop](amas_a2/part1/train.py) implement the independent and centralized approaches.
- [Foraging environment](amas_a2/part2/foraging_env.py) defines the interaction and rewards. The [training loop](amas_a2/part2/train.py) runs the comparison.
- The [report](report.pdf) presents the experiments and interprets the results.
