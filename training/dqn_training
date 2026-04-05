"""
dqn_training.py
─────────────────────────────────────────────────────────────────────────────
DQN (Deep Q-Network) — Value-Based RL
10 hyperparameter experiments on the SRH Education Environment.

Run:
    python training/dqn_training.py

Outputs:
    models/dqn/dqn_best.zip
    models/dqn/results/dqn_hyperparameter_results.csv
    models/dqn/results/dqn_comparison_plot.png
─────────────────────────────────────────────────────────────────────────────
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shutil

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

from environment.custom_env import SRHEducationEnv

BASE_DIR    = os.path.join(os.path.dirname(__file__), "..")
MODEL_DIR   = os.path.join(BASE_DIR, "models", "dqn")
RESULTS_DIR = os.path.join(MODEL_DIR, "results")
LOG_DIR     = os.path.join(MODEL_DIR, "logs")

for d in [MODEL_DIR, RESULTS_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

TOTAL_TIMESTEPS = 50_000

# ── 10 hyperparameter combinations ───────────────────────────────────────
HP_GRID = [
    {"learning_rate": 1e-3, "gamma": 0.99, "batch_size": 64,  "buffer_size": 10000, "exploration_fraction": 0.20, "target_update_interval": 500},
    {"learning_rate": 5e-4, "gamma": 0.99, "batch_size": 64,  "buffer_size": 10000, "exploration_fraction": 0.20, "target_update_interval": 500},
    {"learning_rate": 1e-4, "gamma": 0.99, "batch_size": 64,  "buffer_size": 10000, "exploration_fraction": 0.30, "target_update_interval": 500},
    {"learning_rate": 1e-3, "gamma": 0.95, "batch_size": 32,  "buffer_size":  5000, "exploration_fraction": 0.20, "target_update_interval": 300},
    {"learning_rate": 1e-3, "gamma": 0.90, "batch_size": 128, "buffer_size": 20000, "exploration_fraction": 0.15, "target_update_interval": 1000},
    {"learning_rate": 3e-4, "gamma": 0.99, "batch_size": 64,  "buffer_size": 50000, "exploration_fraction": 0.25, "target_update_interval": 200},
    {"learning_rate": 1e-3, "gamma": 0.99, "batch_size": 256, "buffer_size": 10000, "exploration_fraction": 0.10, "target_update_interval": 500},
    {"learning_rate": 5e-3, "gamma": 0.99, "batch_size": 64,  "buffer_size": 10000, "exploration_fraction": 0.30, "target_update_interval": 500},
    {"learning_rate": 1e-3, "gamma": 0.98, "batch_size": 64,  "buffer_size": 10000, "exploration_fraction": 0.20, "target_update_interval": 750},
    {"learning_rate": 1e-3, "gamma": 0.99, "batch_size": 64,  "buffer_size": 10000, "exploration_fraction": 0.40, "target_update_interval": 500},
]


class EpisodeRewardCallback:
    """Lightweight callback to collect episode rewards."""
    def __init__(self):
        self.episode_rewards = []
        self._cur = 0.0

    def on_step(self, reward, done):
        self._cur += reward
        if done:
            self.episode_rewards.append(self._cur)
            self._cur = 0.0


def plot_dqn_results(all_rewards: list, results_df: pd.DataFrame):
    """Plot cumulative reward curves + objective curve for all 10 DQN runs."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("DQN — Hyperparameter Experiment Results", fontsize=14, fontweight="bold")

    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    # Reward curves per run
    for i, rewards in enumerate(all_rewards):
        if rewards:
            axes[0].plot(rewards, alpha=0.6, color=colors[i],
                         label=f"Run {i+1} (lr={HP_GRID[i]['learning_rate']})")
    axes[0].set_title("Cumulative Reward per Episode")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Total Reward")
    axes[0].legend(fontsize=7, ncol=2)
    axes[0].grid(alpha=0.3)

    # DQN objective curve (mean reward per run)
    mean_rewards = results_df["Mean Reward"].values
    axes[1].bar(range(1, 11), mean_rewards, color=colors)
    axes[1].axhline(np.mean(mean_rewards), color="red", linestyle="--", label="Average")
    axes[1].set_title("DQN Objective Curve (Mean Reward per Run)")
    axes[1].set_xlabel("Run")
    axes[1].set_ylabel("Mean Evaluation Reward")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # Convergence — rolling std of best run
    best_idx = int(results_df["Mean Reward"].idxmax())
    if all_rewards[best_idx]:
        rolling_std = pd.Series(all_rewards[best_idx]).rolling(10).std()
        axes[2].plot(rolling_std, color="darkorange", lw=2)
        axes[2].set_title(f"Convergence Plot — Best Run (Run {best_idx+1})")
        axes[2].set_xlabel("Episode")
        axes[2].set_ylabel("Reward Rolling Std")
        axes[2].grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "dqn_comparison_plot.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  DQN plot saved → {path}")


def train():
    print("=" * 65)
    print("  DQN Training — SRH Education Environment")
    print("  10 Hyperparameter Experiments")
    print("=" * 65)

    rows        = []
    all_rewards = []
    best_mean   = -np.inf
    best_path   = None

    for i, hp in enumerate(HP_GRID):
        run_id  = f"run_{i+1:02d}"
        run_dir = os.path.join(LOG_DIR, run_id)
        os.makedirs(run_dir, exist_ok=True)

        print(f"\n[{run_id}] lr={hp['learning_rate']}  γ={hp['gamma']}  "
              f"batch={hp['batch_size']}  buf={hp['buffer_size']}  "
              f"eps={hp['exploration_fraction']}  tgt={hp['target_update_interval']}")

        train_env = Monitor(SRHEducationEnv(max_steps=200), run_dir)
        eval_env  = SRHEducationEnv(max_steps=200)

        model = DQN(
            policy                 = "MlpPolicy",
            env                    = train_env,
            learning_rate          = hp["learning_rate"],
            gamma                  = hp["gamma"],
            batch_size             = hp["batch_size"],
            buffer_size            = hp["buffer_size"],
            exploration_fraction   = hp["exploration_fraction"],
            target_update_interval = hp["target_update_interval"],
            verbose                = 0,
            seed                   = 42,
        )
        model.learn(total_timesteps=TOTAL_TIMESTEPS)

        mean_r, std_r = evaluate_policy(model, eval_env, n_eval_episodes=20, deterministic=True)
        print(f"  → Mean: {mean_r:.2f} ± {std_r:.2f}")

        # Collect episode rewards from monitor
        from stable_baselines3.common.results_plotter import load_results
        try:
            monitor_data = load_results(run_dir)
            ep_rewards   = list(monitor_data["r"])
        except Exception:
            ep_rewards = []
        all_rewards.append(ep_rewards)

        model_path = os.path.join(MODEL_DIR, f"dqn_{run_id}")
        model.save(model_path)

        rows.append({
            "Run":                    run_id,
            "Learning Rate":          hp["learning_rate"],
            "Gamma":                  hp["gamma"],
            "Batch Size":             hp["batch_size"],
            "Buffer Size":            hp["buffer_size"],
            "Exploration Fraction":   hp["exploration_fraction"],
            "Target Update Interval": hp["target_update_interval"],
            "Mean Reward":            round(mean_r, 3),
            "Std Reward":             round(std_r,  3),
        })

        if mean_r > best_mean:
            best_mean = mean_r
            best_path = model_path

        train_env.close()
        eval_env.close()

    df = pd.DataFrame(rows)
    csv_path = os.path.join(RESULTS_DIR, "dqn_hyperparameter_results.csv")
    df.to_csv(csv_path, index=False)

    plot_dqn_results(all_rewards, df)

    if best_path:
        shutil.copy(f"{best_path}.zip", os.path.join(MODEL_DIR, "dqn_best.zip"))

    print("\n" + "=" * 65)
    print("  DQN Results Table")
    print("=" * 65)
    print(df.to_string(index=False))
    print(f"\n  Best mean reward : {best_mean:.3f}")
    print(f"  CSV saved        → {csv_path}")
    print(f"  Best model       → models/dqn/dqn_best.zip")

    return df


if __name__ == "__main__":
    train()