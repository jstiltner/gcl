#!/usr/bin/env python3
"""
Experiment 01: Single-Agent Commitment Learning

This experiment trains a single agent to learn optimal commitment policies
using PPO (Proximal Policy Optimization). The agent learns to:

1. Commit with high confidence to easy tasks
2. Commit with low confidence to hard tasks
3. Refuse commitment to impossible tasks
4. Calibrate confidence to match actual success probability

Usage:
    python experiments/01_single_agent_commitment_learning.py [--timesteps N] [--seed S]

Example:
    python experiments/01_single_agent_commitment_learning.py --timesteps 50000 --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from gcl.learning.environment import CommitmentEnv, CommitmentEnvConfig, TaskDifficulty
from gcl.learning.policy import CommitmentPolicy, HeuristicPolicy, RandomPolicy
from gcl.learning.reward import CalibrationTracker
from gcl.learning.training import PPOConfig, PPOTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a commitment learning agent with PPO",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=50000,
        help="Total training timesteps (default: 50000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=100,
        help="Number of evaluation episodes (default: 100)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/01_single_agent",
        help="Output directory for results (default: results/01_single_agent)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable plotting (useful for headless environments)",
    )
    return parser.parse_args()


def evaluate_policy_by_difficulty(
    env: CommitmentEnv,
    policy,
    n_episodes: int = 100,
    is_neural: bool = True,
) -> dict:
    """
    Evaluate a policy and break down results by task difficulty.
    
    Args:
        env: The commitment environment.
        policy: Policy to evaluate.
        n_episodes: Number of episodes to run.
        is_neural: Whether the policy is a neural network.
        
    Returns:
        Dictionary with evaluation metrics.
    """
    results = {
        "total_reward": 0.0,
        "total_steps": 0,
        "by_difficulty": {d.value: {"commits": 0, "successes": 0, "abstains": 0, "total": 0} 
                         for d in TaskDifficulty},
        "calibration": CalibrationTracker(),
        "confidences": [],
        "true_probs": [],
    }
    
    for _ in range(n_episodes):
        obs, info = env.reset()
        done = False
        
        while not done:
            # Get action
            if is_neural:
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                action, _, _ = policy.get_action(obs_tensor, deterministic=True)
                action = action[0]
            else:
                action, _, _ = policy.get_action(obs)
            
            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Record results
            results["total_reward"] += reward
            results["total_steps"] += 1
            
            difficulty = info.get("task_difficulty", "medium")
            results["by_difficulty"][difficulty]["total"] += 1
            
            if info.get("outcome") == "abstained":
                results["by_difficulty"][difficulty]["abstains"] += 1
            else:
                results["by_difficulty"][difficulty]["commits"] += 1
                if info.get("outcome") == "success":
                    results["by_difficulty"][difficulty]["successes"] += 1
                
                # Track calibration
                confidence = info.get("action_confidence", 0.5)
                true_prob = info.get("true_success_prob", 0.5)
                success = info.get("outcome") == "success"
                
                results["calibration"].record(confidence, success)
                results["confidences"].append(confidence)
                results["true_probs"].append(true_prob)
            
            obs = next_obs
    
    # Compute summary statistics
    results["mean_reward"] = results["total_reward"] / n_episodes
    results["mean_length"] = results["total_steps"] / n_episodes
    results["calibration_stats"] = results["calibration"].get_statistics()
    
    # Compute commit rates and success rates by difficulty
    for diff in TaskDifficulty:
        d = results["by_difficulty"][diff.value]
        if d["total"] > 0:
            d["commit_rate"] = d["commits"] / d["total"]
            d["success_rate"] = d["successes"] / d["commits"] if d["commits"] > 0 else 0.0
        else:
            d["commit_rate"] = 0.0
            d["success_rate"] = 0.0
    
    return results


def plot_training_curves(metrics, output_path: Path) -> None:
    """Plot training curves."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Episode rewards
    ax = axes[0, 0]
    rewards = metrics.episode_rewards
    ax.plot(rewards, alpha=0.3, label="Episode Reward")
    # Moving average
    window = min(50, len(rewards) // 10) if len(rewards) > 10 else 1
    if window > 1:
        ma = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(rewards)), ma, label=f"MA({window})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("Training Rewards")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Episode lengths
    ax = axes[0, 1]
    lengths = metrics.episode_lengths
    ax.plot(lengths, alpha=0.3, label="Episode Length")
    if window > 1:
        ma = np.convolve(lengths, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(lengths)), ma, label=f"MA({window})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Length")
    ax.set_title("Episode Lengths")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Policy loss
    ax = axes[1, 0]
    if metrics.policy_losses:
        ax.plot(metrics.policy_losses)
        ax.set_xlabel("Update")
        ax.set_ylabel("Loss")
        ax.set_title("Policy Loss")
        ax.grid(True, alpha=0.3)
    
    # Value loss
    ax = axes[1, 1]
    if metrics.value_losses:
        ax.plot(metrics.value_losses)
        ax.set_xlabel("Update")
        ax.set_ylabel("Loss")
        ax.set_title("Value Loss")
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / "training_curves.png", dpi=150)
    plt.close()


def plot_calibration(results: dict, output_path: Path, title: str = "Calibration") -> None:
    """Plot calibration curve."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Calibration curve
    ax = axes[0]
    curve = results["calibration"].get_calibration_curve()
    if curve:
        # Curve is list of (expected_conf, actual_rate, count) tuples
        bins = [c[0] for c in curve]
        actual = [c[1] for c in curve]
        counts = [c[2] for c in curve]
        
        # Plot calibration
        ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
        ax.scatter(bins, actual, s=[max(c*2, 10) for c in counts], alpha=0.7, label="Actual")
        ax.set_xlabel("Predicted Confidence")
        ax.set_ylabel("Actual Success Rate")
        ax.set_title(f"{title} - Calibration Curve")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    
    # Confidence vs True Probability scatter
    ax = axes[1]
    if results["confidences"] and results["true_probs"]:
        ax.scatter(results["true_probs"], results["confidences"], alpha=0.3, s=10)
        ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
        ax.set_xlabel("True Success Probability")
        ax.set_ylabel("Agent Confidence")
        ax.set_title(f"{title} - Confidence vs True Probability")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(output_path / f"calibration_{title.lower().replace(' ', '_')}.png", dpi=150)
    plt.close()


def plot_difficulty_comparison(
    trained_results: dict,
    random_results: dict,
    heuristic_results: dict,
    output_path: Path,
) -> None:
    """Plot comparison of policies by difficulty."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    difficulties = [d.value for d in TaskDifficulty]
    x = np.arange(len(difficulties))
    width = 0.25
    
    # Commit rates
    ax = axes[0]
    trained_commits = [trained_results["by_difficulty"][d]["commit_rate"] for d in difficulties]
    random_commits = [random_results["by_difficulty"][d]["commit_rate"] for d in difficulties]
    heuristic_commits = [heuristic_results["by_difficulty"][d]["commit_rate"] for d in difficulties]
    
    ax.bar(x - width, trained_commits, width, label="Trained", color="blue", alpha=0.7)
    ax.bar(x, random_commits, width, label="Random", color="gray", alpha=0.7)
    ax.bar(x + width, heuristic_commits, width, label="Heuristic", color="green", alpha=0.7)
    ax.set_xlabel("Task Difficulty")
    ax.set_ylabel("Commit Rate")
    ax.set_title("Commit Rate by Difficulty")
    ax.set_xticks(x)
    ax.set_xticklabels(difficulties)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Success rates (among commits)
    ax = axes[1]
    trained_success = [trained_results["by_difficulty"][d]["success_rate"] for d in difficulties]
    random_success = [random_results["by_difficulty"][d]["success_rate"] for d in difficulties]
    heuristic_success = [heuristic_results["by_difficulty"][d]["success_rate"] for d in difficulties]
    
    ax.bar(x - width, trained_success, width, label="Trained", color="blue", alpha=0.7)
    ax.bar(x, random_success, width, label="Random", color="gray", alpha=0.7)
    ax.bar(x + width, heuristic_success, width, label="Heuristic", color="green", alpha=0.7)
    ax.set_xlabel("Task Difficulty")
    ax.set_ylabel("Success Rate")
    ax.set_title("Success Rate by Difficulty (among commits)")
    ax.set_xticks(x)
    ax.set_xticklabels(difficulties)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Overall comparison
    ax = axes[2]
    metrics = ["Mean Reward", "ECE (lower=better)"]
    trained_vals = [
        trained_results["mean_reward"],
        trained_results["calibration_stats"]["expected_calibration_error"],
    ]
    random_vals = [
        random_results["mean_reward"],
        random_results["calibration_stats"]["expected_calibration_error"],
    ]
    heuristic_vals = [
        heuristic_results["mean_reward"],
        heuristic_results["calibration_stats"]["expected_calibration_error"],
    ]
    
    x = np.arange(len(metrics))
    ax.bar(x - width, trained_vals, width, label="Trained", color="blue", alpha=0.7)
    ax.bar(x, random_vals, width, label="Random", color="gray", alpha=0.7)
    ax.bar(x + width, heuristic_vals, width, label="Heuristic", color="green", alpha=0.7)
    ax.set_xlabel("Metric")
    ax.set_ylabel("Value")
    ax.set_title("Overall Performance Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path / "difficulty_comparison.png", dpi=150)
    plt.close()


def main() -> None:
    """Run the experiment."""
    args = parse_args()
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting experiment with seed={args.seed}, timesteps={args.timesteps}")
    logger.info(f"Output directory: {output_path}")
    
    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create environment
    env_config = CommitmentEnvConfig(
        feature_dim=8,
        max_steps=100,
        initial_stake=100.0,
        calibration_bonus=0.2,  # Encourage calibration
    )
    env = CommitmentEnv(config=env_config, seed=args.seed)
    
    # Create PPO config
    ppo_config = PPOConfig(
        total_timesteps=args.timesteps,
        n_steps=256,
        batch_size=64,
        n_epochs=10,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        log_interval=10,
    )
    
    # Create trainer
    trainer = PPOTrainer(
        env=env,
        config=ppo_config,
        device="cuda" if torch.cuda.is_available() else "cpu",
        seed=args.seed,
    )
    
    logger.info("Starting training...")
    start_time = datetime.now()
    
    # Train
    metrics = trainer.train()
    
    training_time = (datetime.now() - start_time).total_seconds()
    logger.info(f"Training completed in {training_time:.1f} seconds")
    logger.info(f"Total timesteps: {metrics.timesteps}")
    logger.info(f"Total episodes: {metrics.episodes}")
    
    # Evaluate trained policy
    logger.info("Evaluating trained policy...")
    eval_env = CommitmentEnv(config=env_config, seed=args.seed + 1000)
    trained_results = evaluate_policy_by_difficulty(
        eval_env, trainer.policy, n_episodes=args.eval_episodes, is_neural=True
    )
    
    # Evaluate baselines
    logger.info("Evaluating random baseline...")
    random_policy = RandomPolicy(seed=args.seed)
    eval_env = CommitmentEnv(config=env_config, seed=args.seed + 1000)
    random_results = evaluate_policy_by_difficulty(
        eval_env, random_policy, n_episodes=args.eval_episodes, is_neural=False
    )
    
    logger.info("Evaluating heuristic baseline...")
    heuristic_policy = HeuristicPolicy()
    eval_env = CommitmentEnv(config=env_config, seed=args.seed + 1000)
    heuristic_results = evaluate_policy_by_difficulty(
        eval_env, heuristic_policy, n_episodes=args.eval_episodes, is_neural=False
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("EXPERIMENT RESULTS")
    print("=" * 60)
    
    print("\n--- Trained Policy ---")
    print(f"Mean Reward: {trained_results['mean_reward']:.3f}")
    print(f"Mean Episode Length: {trained_results['mean_length']:.1f}")
    print(f"ECE: {trained_results['calibration_stats']['expected_calibration_error']:.4f}")
    
    print("\n--- Random Baseline ---")
    print(f"Mean Reward: {random_results['mean_reward']:.3f}")
    print(f"Mean Episode Length: {random_results['mean_length']:.1f}")
    print(f"ECE: {random_results['calibration_stats']['expected_calibration_error']:.4f}")
    
    print("\n--- Heuristic Baseline ---")
    print(f"Mean Reward: {heuristic_results['mean_reward']:.3f}")
    print(f"Mean Episode Length: {heuristic_results['mean_length']:.1f}")
    print(f"ECE: {heuristic_results['calibration_stats']['expected_calibration_error']:.4f}")
    
    print("\n--- Commit Rates by Difficulty ---")
    print(f"{'Difficulty':<12} {'Trained':>10} {'Random':>10} {'Heuristic':>10}")
    for diff in TaskDifficulty:
        t = trained_results["by_difficulty"][diff.value]["commit_rate"]
        r = random_results["by_difficulty"][diff.value]["commit_rate"]
        h = heuristic_results["by_difficulty"][diff.value]["commit_rate"]
        print(f"{diff.value:<12} {t:>10.2%} {r:>10.2%} {h:>10.2%}")
    
    print("\n--- Success Rates by Difficulty (among commits) ---")
    print(f"{'Difficulty':<12} {'Trained':>10} {'Random':>10} {'Heuristic':>10}")
    for diff in TaskDifficulty:
        t = trained_results["by_difficulty"][diff.value]["success_rate"]
        r = random_results["by_difficulty"][diff.value]["success_rate"]
        h = heuristic_results["by_difficulty"][diff.value]["success_rate"]
        print(f"{diff.value:<12} {t:>10.2%} {r:>10.2%} {h:>10.2%}")
    
    # Save results
    results_data = {
        "config": {
            "timesteps": args.timesteps,
            "seed": args.seed,
            "eval_episodes": args.eval_episodes,
        },
        "training": {
            "time_seconds": training_time,
            "total_timesteps": metrics.timesteps,
            "total_episodes": metrics.episodes,
            "final_reward": float(np.mean(metrics.episode_rewards[-100:])) if metrics.episode_rewards else 0,
        },
        "trained": {
            "mean_reward": trained_results["mean_reward"],
            "mean_length": trained_results["mean_length"],
            "ece": trained_results["calibration_stats"]["expected_calibration_error"],
            "by_difficulty": {
                d: {k: v for k, v in trained_results["by_difficulty"][d].items() if k != "calibration"}
                for d in trained_results["by_difficulty"]
            },
        },
        "random": {
            "mean_reward": random_results["mean_reward"],
            "mean_length": random_results["mean_length"],
            "ece": random_results["calibration_stats"]["expected_calibration_error"],
        },
        "heuristic": {
            "mean_reward": heuristic_results["mean_reward"],
            "mean_length": heuristic_results["mean_length"],
            "ece": heuristic_results["calibration_stats"]["expected_calibration_error"],
        },
    }
    
    with open(output_path / "results.json", "w") as f:
        json.dump(results_data, f, indent=2)
    
    logger.info(f"Results saved to {output_path / 'results.json'}")
    
    # Save model
    torch.save(trainer.policy.state_dict(), output_path / "policy.pt")
    logger.info(f"Model saved to {output_path / 'policy.pt'}")
    
    # Generate plots
    if not args.no_plot:
        logger.info("Generating plots...")
        
        plot_training_curves(metrics, output_path)
        plot_calibration(trained_results, output_path, "Trained Policy")
        plot_calibration(random_results, output_path, "Random Policy")
        plot_calibration(heuristic_results, output_path, "Heuristic Policy")
        plot_difficulty_comparison(
            trained_results, random_results, heuristic_results, output_path
        )
        
        logger.info(f"Plots saved to {output_path}")
    
    print("\n" + "=" * 60)
    print(f"Experiment complete! Results saved to {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
