"""
AMAS Assignment 2 - Main Entry Point

Runs all experiments and generates visualizations:
1. Part 1: Train IQL and CQL on Pursuit, compare performance
2. Part 2: Train IQL on Level-Based Foraging with global vs mixed rewards

Usage:
    python run_all.py [--quick] [--device cpu]
"""

import os
import sys
import argparse
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
import json
import pickle
import time

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from amas_a2.part1.train import train_iql, train_cql
from amas_a2.part2.train import train_iql_foraging


FIGURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def smooth_curve(data, window=11):
    """Apply moving average smoothing."""
    if len(data) < window:
        return data
    return uniform_filter1d(data, size=window, mode="nearest")


def plot_training_curves(iql_results, cql_results, save_path):
    """Plot Part 1: IQL vs CQL training curves with smoothed avg over seeds."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax_idx, (label, results_dict, color) in enumerate(
        [
            ("IQL", iql_results, "steelblue"),
            ("CQL", cql_results, "coral"),
        ]
    ):
        ax = axes[ax_idx]

        # Plot individual seed runs
        for seed, data in results_dict.items():
            eps = data["eval_episodes"]
            rewards = data["eval_rewards"]
            ax.plot(eps, rewards, alpha=0.3, color=color, linewidth=0.8)

        # Compute mean and std across seeds
        all_eps = None
        all_rewards = []
        for seed, data in results_dict.items():
            if all_eps is None:
                all_eps = data["eval_episodes"]
            all_rewards.append(data["eval_rewards"])

        mean_rewards = np.mean(all_rewards, axis=0)
        std_rewards = np.std(all_rewards, axis=0)

        # Smooth
        smooth_mean = smooth_curve(mean_rewards)
        smooth_std = smooth_curve(std_rewards)

        ax.plot(all_eps, smooth_mean, color=color, linewidth=2.5, label=f"{label} (mean)")
        ax.fill_between(
            all_eps,
            smooth_mean - smooth_std,
            smooth_mean + smooth_std,
            color=color,
            alpha=0.15,
        )

        ax.set_xlabel("Episode", fontsize=12)
        ax.set_ylabel("Cumulative Reward", fontsize=12)
        ax.set_title(f"{label} on Pursuit", fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        "Part 1: IQL vs CQL on Pursuit Environment",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_iql_vs_cql_comparison(iql_results, cql_results, save_path):
    """Direct comparison of IQL vs CQL on same plot."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {"IQL": "steelblue", "CQL": "coral"}

    for label, results_dict in [("IQL", iql_results), ("CQL", cql_results)]:
        all_rewards = []
        all_eps = None
        for seed, data in results_dict.items():
            if all_eps is None:
                all_eps = data["eval_episodes"]
            all_rewards.append(data["eval_rewards"])

        mean_rewards = np.mean(all_rewards, axis=0)
        std_rewards = np.std(all_rewards, axis=0)
        smooth_mean = smooth_curve(mean_rewards)
        smooth_std = smooth_curve(std_rewards)

        ax.plot(
            all_eps, smooth_mean, color=colors[label], linewidth=2.5, label=label
        )
        ax.fill_between(
            all_eps,
            smooth_mean - smooth_std,
            smooth_mean + smooth_std,
            color=colors[label],
            alpha=0.2,
        )

    ax.set_xlabel("Episode", fontsize=13)
    ax.set_ylabel("Cumulative Reward", fontsize=13)
    ax.set_title("IQL vs CQL: Performance Comparison on Pursuit", fontsize=15)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_foraging_comparison(global_results, mixed_results, save_path):
    """Plot Part 2: Global vs Mixed reward comparison."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {"Global Reward": "forestgreen", "Mixed Reward": "darkorange"}

    for label, results_dict in [
        ("Global Reward", global_results),
        ("Mixed Reward", mixed_results),
    ]:
        all_rewards = []
        all_eps = None
        for seed, data in results_dict.items():
            if all_eps is None:
                all_eps = data["eval_episodes"]
            all_rewards.append(data["eval_rewards"])

        mean_rewards = np.mean(all_rewards, axis=0)
        std_rewards = np.std(all_rewards, axis=0)
        smooth_mean = smooth_curve(mean_rewards)
        smooth_std = smooth_curve(std_rewards)

        ax.plot(
            all_eps, smooth_mean, color=colors[label], linewidth=2.5, label=label
        )
        ax.fill_between(
            all_eps,
            smooth_mean - smooth_std,
            smooth_mean + smooth_std,
            color=colors[label],
            alpha=0.2,
        )

    ax.set_xlabel("Episode", fontsize=13)
    ax.set_ylabel("Cumulative Reward", fontsize=13)
    ax.set_title(
        "Part 2: Global vs Mixed Reward on Level-Based Foraging",
        fontsize=15,
    )
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def run_part1(device="cpu", quick=False, fast=False):
    """Run Part 1 experiments."""
    print("=" * 60)
    print("PART 1: IQL vs CQL on Pursuit")
    print("=" * 60)

    if fast:
        n_episodes = 30
        n_seeds = 3
        eval_interval = 15
        eval_episodes = 1
    elif quick:
        n_episodes = 500
        n_seeds = 3
        eval_interval = 50
        eval_episodes = 3
    else:
        n_episodes = 2000
        n_seeds = 3
        eval_interval = 50
        eval_episodes = 3

    env_config = {}  # Use defaults

    iql_results = {}
    cql_results = {}

    # Train IQL with multiple seeds
    print("\n--- Training IQL ---")
    for seed in range(n_seeds):
        print(f"\nSeed {seed + 1}/{n_seeds}")
        result = train_iql(
            env_config=env_config,
            n_episodes=n_episodes,
            eval_interval=eval_interval,
            eval_episodes=eval_episodes,
            lr=1e-4,
            gamma=0.99,
            batch_size=64,
            target_update_freq=500,
            epsilon_decay=0.999,
            device=device,
            seed=seed + 42,
            verbose=True,
        )
        iql_results[seed] = result

    # Train CQL with multiple seeds
    print("\n--- Training CQL ---")
    for seed in range(n_seeds):
        print(f"\nSeed {seed + 1}/{n_seeds}")
        result = train_cql(
            env_config=env_config,
            n_episodes=n_episodes,
            eval_interval=eval_interval,
            eval_episodes=eval_episodes,
            lr=1e-4,
            gamma=0.99,
            batch_size=64,
            target_update_freq=500,
            epsilon_decay=0.999,
            device=device,
            seed=seed + 42,
            verbose=True,
        )
        cql_results[seed] = result

    result_tag = "fast" if fast else "quick" if quick else "full"

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, f"part1_results_{result_tag}.pkl"), "wb") as f:
        pickle.dump({"iql": iql_results, "cql": cql_results}, f)

    # Generate plots
    os.makedirs(FIGURE_DIR, exist_ok=True)
    plot_training_curves(
        iql_results,
        cql_results,
        FIGURE_DIR + os.sep + f"part1_iql_cql_training_{result_tag}.png",
    )
    plot_iql_vs_cql_comparison(
        iql_results,
        cql_results,
        FIGURE_DIR + os.sep + f"part1_iql_vs_cql_comparison_{result_tag}.png",
    )

    return iql_results, cql_results


def run_part2(device="cpu", quick=False, fast=False):
    """Run Part 2 experiments."""
    print("\n" + "=" * 60)
    print("PART 2: Level-Based Foraging with Different Rewards")
    print("=" * 60)

    if fast:
        n_episodes = 40
        n_seeds = 3
        eval_interval = 15
        eval_episodes = 1
    elif quick:
        n_episodes = 500
        n_seeds = 3
        eval_interval = 100
        eval_episodes = 3
    else:
        n_episodes = 3000
        n_seeds = 3
        eval_interval = 100
        eval_episodes = 5

    global_results = {}
    mixed_results = {}

    # Global reward
    print("\n--- Training with Global Reward ---")
    for seed in range(n_seeds):
        print(f"\nSeed {seed + 1}/{n_seeds}")
        result = train_iql_foraging(
            reward_fn="global",
            grid_size=8,
            n_agents=3,
            n_crops=5,
            max_level=5,
            max_steps=150,
            static=False,
            n_episodes=n_episodes,
            eval_interval=eval_interval,
            eval_episodes=eval_episodes,
            lr=1e-4,
            gamma=0.99,
            batch_size=64,
            target_update_freq=500,
            epsilon_decay=0.999,
            device=device,
            seed=seed + 42,
            verbose=True,
        )
        global_results[seed] = result

    # Mixed reward
    print("\n--- Training with Mixed Reward ---")
    for seed in range(n_seeds):
        print(f"\nSeed {seed + 1}/{n_seeds}")
        result = train_iql_foraging(
            reward_fn="mixed",
            grid_size=8,
            n_agents=3,
            n_crops=5,
            max_level=5,
            max_steps=150,
            static=False,
            n_episodes=n_episodes,
            eval_interval=eval_interval,
            eval_episodes=eval_episodes,
            lr=1e-4,
            gamma=0.99,
            batch_size=64,
            target_update_freq=500,
            epsilon_decay=0.999,
            device=device,
            seed=seed + 42,
            verbose=True,
        )
        mixed_results[seed] = result

    result_tag = "fast" if fast else "quick" if quick else "full"

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, f"part2_results_{result_tag}.pkl"), "wb") as f:
        pickle.dump({"global": global_results, "mixed": mixed_results}, f)

    # Generate plots
    os.makedirs(FIGURE_DIR, exist_ok=True)
    plot_foraging_comparison(
        global_results,
        mixed_results,
        FIGURE_DIR + os.sep + f"part2_foraging_reward_comparison_{result_tag}.png",
    )

    return global_results, mixed_results


def main():
    parser = argparse.ArgumentParser(description="AMAS Assignment 2")
    parser.add_argument("--quick", action="store_true", help="Quick test run (fewer episodes)")
    parser.add_argument("--fast", action="store_true", help="Faster demo run (~1 hour)")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device to use")
    args = parser.parse_args()

    os.makedirs(FIGURE_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    start_time = time.time()

    # Part 1
    try:
        iql_results, cql_results = run_part1(args.device, args.quick, args.fast)
    except Exception as e:
        print(f"Part 1 failed: {e}")
        import traceback
        traceback.print_exc()

    # Part 2
    try:
        global_results, mixed_results = run_part2(args.device, args.quick, args.fast)
    except Exception as e:
        print(f"Part 2 failed: {e}")
        import traceback
        traceback.print_exc()

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"All experiments completed in {elapsed / 60:.1f} minutes")
    print(f"Results saved to: {RESULTS_DIR}")
    print(f"Figures saved to: {FIGURE_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
