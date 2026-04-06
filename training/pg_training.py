"""
pg_training.py
─────────────────────────────────────────────────────────────────────────────
Policy Gradient Methods: REINFORCE (custom PyTorch), PPO (Stable Baselines 3)
10 hyperparameter experiments each on the SRH Education Environment.
NOTE: A2C has been removed from this assignment.

Run:
    python training/pg_training.py

Outputs:
    models/pg/reinforce_best.pt       <- best REINFORCE model weights
    models/pg/ppo_best.zip            <- best PPO model
    models/pg/results/reinforce_results.csv
    models/pg/results/ppo_results.csv
    models/pg/results/pg_comparison_plot.png
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

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.results_plotter import load_results

from environment.custom_env import SRHEducationEnv

BASE_DIR    = os.path.join(os.path.dirname(__file__), "..")
MODEL_DIR   = os.path.join(BASE_DIR, "models", "pg")
RESULTS_DIR = os.path.join(MODEL_DIR, "results")
LOG_DIR     = os.path.join(MODEL_DIR, "logs")

for d in [MODEL_DIR, RESULTS_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

TOTAL_TIMESTEPS = 50_000


# ============================================================
# REINFORCE — Custom PyTorch implementation
# ============================================================

class PolicyNet(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),  nn.ReLU(),
            nn.Linear(hidden, action_dim), nn.Softmax(dim=-1),
        )

    def forward(self, x):
        return self.net(x)


def train_reinforce_run(hp: dict, run_id: str):
    lr           = hp["learning_rate"]
    gamma        = hp["gamma"]
    n_episodes   = hp.get("n_episodes", 500)
    entropy_coef = hp.get("entropy_coef", 0.01)

    env        = SRHEducationEnv(max_steps=200)
    obs_dim    = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy    = PolicyNet(obs_dim, action_dim)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    ep_rewards  = []
    entropy_log = []

    for ep in range(n_episodes):
        obs, _   = env.reset()
        log_probs, rewards_ep, entropies = [], [], []
        done = False

        while not done:
            t      = torch.FloatTensor(obs).unsqueeze(0)
            probs  = policy(t)
            dist   = Categorical(probs)
            action = dist.sample()
            log_probs.append(dist.log_prob(action))
            entropies.append(dist.entropy())
            obs, r, terminated, truncated, _ = env.step(action.item())
            rewards_ep.append(r)
            done = terminated or truncated

        # Discounted returns
        G, returns = 0.0, []
        for r in reversed(rewards_ep):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns)
        if returns.std() > 1e-8:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        loss = sum(-lp * R - entropy_coef * ent
                   for lp, R, ent in zip(log_probs, returns, entropies))
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()

        ep_rewards.append(sum(rewards_ep))
        entropy_log.append(float(torch.stack(entropies).mean()))

    env.close()

    # Evaluate
    eval_env = SRHEducationEnv(max_steps=200)
    policy.eval()
    eval_rs = []
    with torch.no_grad():
        for _ in range(20):
            obs, _ = eval_env.reset()
            total, done = 0.0, False
            while not done:
                probs  = policy(torch.FloatTensor(obs).unsqueeze(0))
                action = probs.argmax().item()
                obs, r, t, tr, _ = eval_env.step(action)
                total += r
                done   = t or tr
            eval_rs.append(total)
    eval_env.close()

    # Save this run
    save_path = os.path.join(MODEL_DIR, f"reinforce_{run_id}.pt")
    torch.save(policy.state_dict(), save_path)
    print(f"  Model saved -> {save_path}")

    return {
        "ep_rewards":  ep_rewards,
        "entropy_log": entropy_log,
        "mean_reward": float(np.mean(eval_rs)),
        "std_reward":  float(np.std(eval_rs)),
        "model_path":  save_path,
    }


# Hyperparameter grids
REINFORCE_GRID = [
    {"learning_rate": 1e-3, "gamma": 0.99, "n_episodes": 500, "entropy_coef": 0.010},
    {"learning_rate": 5e-4, "gamma": 0.99, "n_episodes": 500, "entropy_coef": 0.010},
    {"learning_rate": 1e-4, "gamma": 0.99, "n_episodes": 500, "entropy_coef": 0.010},
    {"learning_rate": 1e-3, "gamma": 0.95, "n_episodes": 500, "entropy_coef": 0.010},
    {"learning_rate": 1e-3, "gamma": 0.90, "n_episodes": 500, "entropy_coef": 0.010},
    {"learning_rate": 1e-3, "gamma": 0.99, "n_episodes": 300, "entropy_coef": 0.010},
    {"learning_rate": 1e-3, "gamma": 0.99, "n_episodes": 800, "entropy_coef": 0.010},
    {"learning_rate": 1e-3, "gamma": 0.99, "n_episodes": 500, "entropy_coef": 0.050},
    {"learning_rate": 1e-3, "gamma": 0.99, "n_episodes": 500, "entropy_coef": 0.001},
    {"learning_rate": 3e-4, "gamma": 0.98, "n_episodes": 600, "entropy_coef": 0.020},
]

PPO_GRID = [
    {"learning_rate": 3e-4, "gamma": 0.99, "n_steps": 2048, "ent_coef": 0.00, "clip_range": 0.20},
    {"learning_rate": 1e-4, "gamma": 0.99, "n_steps": 2048, "ent_coef": 0.00, "clip_range": 0.20},
    {"learning_rate": 1e-3, "gamma": 0.99, "n_steps": 2048, "ent_coef": 0.01, "clip_range": 0.20},
    {"learning_rate": 3e-4, "gamma": 0.95, "n_steps": 1024, "ent_coef": 0.00, "clip_range": 0.20},
    {"learning_rate": 3e-4, "gamma": 0.99, "n_steps":  512, "ent_coef": 0.00, "clip_range": 0.10},
    {"learning_rate": 3e-4, "gamma": 0.99, "n_steps": 2048, "ent_coef": 0.00, "clip_range": 0.30},
    {"learning_rate": 5e-4, "gamma": 0.98, "n_steps": 2048, "ent_coef": 0.01, "clip_range": 0.20},
    {"learning_rate": 3e-4, "gamma": 0.99, "n_steps": 4096, "ent_coef": 0.00, "clip_range": 0.20},
    {"learning_rate": 1e-4, "gamma": 0.99, "n_steps": 2048, "ent_coef": 0.05, "clip_range": 0.20},
    {"learning_rate": 3e-4, "gamma": 0.90, "n_steps": 2048, "ent_coef": 0.00, "clip_range": 0.20},
]


# ============================================================
# Train REINFORCE — 10 runs, saves best
# ============================================================

def train_reinforce():
    print("\n" + "=" * 65)
    print("  REINFORCE Training — 10 Hyperparameter Runs")
    print("=" * 65)

    rows, all_ep_rewards, all_entropy = [], [], []
    best_mean, best_path = -np.inf, None

    for i, hp in enumerate(REINFORCE_GRID):
        run_id = f"run_{i+1:02d}"
        print(f"\n[{run_id}] lr={hp['learning_rate']}  "
              f"gamma={hp['gamma']}  "
              f"episodes={hp['n_episodes']}  "
              f"entropy_coef={hp['entropy_coef']}")

        out = train_reinforce_run(hp, run_id)
        print(f"  -> Mean reward: {out['mean_reward']:.2f} "
              f"+/- {out['std_reward']:.2f}")

        all_ep_rewards.append(out["ep_rewards"])
        all_entropy.append(out["entropy_log"])

        rows.append({
            "Run":           run_id,
            "Learning Rate": hp["learning_rate"],
            "Gamma":         hp["gamma"],
            "N Episodes":    hp["n_episodes"],
            "Entropy Coef":  hp["entropy_coef"],
            "Mean Reward":   round(out["mean_reward"], 3),
            "Std Reward":    round(out["std_reward"],  3),
        })

        if out["mean_reward"] > best_mean:
            best_mean = out["mean_reward"]
            best_path = out["model_path"]
            best_dest = os.path.join(MODEL_DIR, "reinforce_best.pt")
            shutil.copy(best_path, best_dest)
            print(f"  BEST so far! Copied -> models/pg/reinforce_best.pt")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "reinforce_results.csv"), index=False)

    print("\n" + "=" * 65)
    print("  REINFORCE Results Table")
    print("=" * 65)
    print(df.to_string(index=False))
    print(f"\n  Best mean reward : {best_mean:.3f}")
    print(f"  Best model       -> models/pg/reinforce_best.pt")

    return df, all_ep_rewards, all_entropy, best_mean


# ============================================================
# Train PPO — 10 runs, saves best
# ============================================================

def train_ppo():
    print("\n" + "=" * 65)
    print("  PPO Training — 10 Hyperparameter Runs")
    print("=" * 65)

    rows, all_ep_rewards = [], []
    best_mean, best_path = -np.inf, None

    for i, hp in enumerate(PPO_GRID):
        run_id  = f"run_{i+1:02d}"
        run_dir = os.path.join(LOG_DIR, f"ppo_{run_id}")
        os.makedirs(run_dir, exist_ok=True)

        print(f"\n[{run_id}] lr={hp['learning_rate']}  "
              f"gamma={hp['gamma']}  "
              f"n_steps={hp['n_steps']}  "
              f"ent_coef={hp['ent_coef']}  "
              f"clip={hp['clip_range']}")

        train_env = Monitor(SRHEducationEnv(max_steps=200), run_dir)
        eval_env  = SRHEducationEnv(max_steps=200)

        model = PPO(
            policy        = "MlpPolicy",
            env           = train_env,
            learning_rate = hp["learning_rate"],
            gamma         = hp["gamma"],
            n_steps       = hp["n_steps"],
            ent_coef      = hp["ent_coef"],
            clip_range    = hp["clip_range"],
            verbose       = 0,
            seed          = 42,
        )
        model.learn(total_timesteps=TOTAL_TIMESTEPS)

        mean_r, std_r = evaluate_policy(
            model, eval_env, n_eval_episodes=20, deterministic=True
        )
        print(f"  -> Mean reward: {mean_r:.2f} +/- {std_r:.2f}")

        try:
            monitor_data = load_results(run_dir)
            ep_rewards   = list(monitor_data["r"])
        except Exception:
            ep_rewards = []
        all_ep_rewards.append(ep_rewards)

        model_path = os.path.join(MODEL_DIR, f"ppo_{run_id}")
        model.save(model_path)
        print(f"  Model saved -> {model_path}.zip")

        rows.append({
            "Run":           run_id,
            "Learning Rate": hp["learning_rate"],
            "Gamma":         hp["gamma"],
            "N Steps":       hp["n_steps"],
            "Ent Coef":      hp["ent_coef"],
            "Clip Range":    hp["clip_range"],
            "Mean Reward":   round(mean_r, 3),
            "Std Reward":    round(std_r,  3),
        })

        if mean_r > best_mean:
            best_mean = mean_r
            best_path = model_path
            best_dest = os.path.join(MODEL_DIR, "ppo_best.zip")
            shutil.copy(f"{model_path}.zip", best_dest)
            print(f"  BEST so far! Copied -> models/pg/ppo_best.zip")

        train_env.close()
        eval_env.close()

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "ppo_results.csv"), index=False)

    print("\n" + "=" * 65)
    print("  PPO Results Table")
    print("=" * 65)
    print(df.to_string(index=False))
    print(f"\n  Best mean reward : {best_mean:.3f}")
    print(f"  Best model       -> models/pg/ppo_best.zip")

    return df, all_ep_rewards, best_mean


# ============================================================
# Combined comparison plots (required by rubric)
# ============================================================

def plot_all_results(r_df, p_df, r_rewards, p_rewards, r_entropy):
    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    fig.suptitle(
        "Policy Gradient Methods — SRH Education RL\n"
        "REINFORCE vs PPO Hyperparameter Experiments",
        fontsize=14, fontweight="bold"
    )
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    # REINFORCE cumulative reward curves
    for i, rewards in enumerate(r_rewards):
        if rewards:
            axes[0, 0].plot(rewards, alpha=0.5, color=colors[i],
                            label=f"Run {i+1}")
    axes[0, 0].set_title("REINFORCE — Cumulative Reward per Episode")
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Total Reward")
    axes[0, 0].legend(fontsize=7, ncol=2)
    axes[0, 0].grid(alpha=0.3)

    # REINFORCE entropy curves
    for i, ent in enumerate(r_entropy):
        if ent:
            axes[0, 1].plot(ent, alpha=0.5, color=colors[i],
                            label=f"Run {i+1}")
    axes[0, 1].set_title("REINFORCE — Policy Entropy Curves")
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Mean Policy Entropy")
    axes[0, 1].legend(fontsize=7, ncol=2)
    axes[0, 1].grid(alpha=0.3)

    # PPO cumulative reward curves
    for i, rewards in enumerate(p_rewards):
        if rewards:
            axes[0, 2].plot(rewards, alpha=0.5, color=colors[i],
                            label=f"Run {i+1}")
    axes[0, 2].set_title("PPO — Cumulative Reward per Episode")
    axes[0, 2].set_xlabel("Episode")
    axes[0, 2].set_ylabel("Total Reward")
    axes[0, 2].legend(fontsize=7, ncol=2)
    axes[0, 2].grid(alpha=0.3)

    # Convergence plot — best run per algorithm
    best_r_idx = int(r_df["Mean Reward"].idxmax()) if not r_df.empty else 0
    best_p_idx = int(p_df["Mean Reward"].idxmax()) if not p_df.empty else 0
    for label, rewards_list, idx, col in [
        ("REINFORCE", r_rewards, best_r_idx, "steelblue"),
        ("PPO",       p_rewards, best_p_idx, "darkorange"),
    ]:
        if rewards_list and len(rewards_list) > idx and rewards_list[idx]:
            rolling = pd.Series(rewards_list[idx]).rolling(10).std()
            axes[1, 0].plot(rolling, label=f"{label} best run",
                            color=col, lw=2)
    axes[1, 0].set_title("Convergence Plot — Best Run per Algorithm")
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_ylabel("Reward Rolling Std (window=10)")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    # REINFORCE mean reward per run
    axes[1, 1].bar(range(1, 11), r_df["Mean Reward"].values,
                   color=colors, alpha=0.85)
    axes[1, 1].axhline(r_df["Mean Reward"].mean(), color="red",
                       linestyle="--", label="Average")
    axes[1, 1].set_title("REINFORCE — Mean Reward per Run")
    axes[1, 1].set_xlabel("Run")
    axes[1, 1].set_ylabel("Mean Evaluation Reward")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3, axis="y")

    # Generalisation comparison bar
    algos      = ["REINFORCE", "PPO"]
    means      = [r_df["Mean Reward"].max(), p_df["Mean Reward"].max()]
    stds       = [
        float(r_df.loc[r_df["Mean Reward"].idxmax(), "Std Reward"]),
        float(p_df.loc[p_df["Mean Reward"].idxmax(), "Std Reward"]),
    ]
    bars = axes[1, 2].bar(algos, means, yerr=stds,
                          color=["steelblue", "darkorange"],
                          capsize=8, alpha=0.85)
    for bar, mean in zip(bars, means):
        axes[1, 2].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 5,
            f"{mean:.1f}", ha="center", va="bottom",
            fontsize=11, fontweight="bold"
        )
    axes[1, 2].set_title("Generalisation Test — Best Model per Algorithm")
    axes[1, 2].set_ylabel("Mean Evaluation Reward +/- Std")
    axes[1, 2].grid(alpha=0.3, axis="y")

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "pg_comparison_plot.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\n  PG comparison plot saved -> {path}")


# ============================================================
# Entry point
# ============================================================

def train():
    r_df, r_rewards, r_entropy, r_best = train_reinforce()
    p_df, p_rewards, p_best            = train_ppo()

    plot_all_results(r_df, p_df, r_rewards, p_rewards, r_entropy)

    print("\n" + "=" * 65)
    print("  Policy Gradient Training Complete")
    print("=" * 65)
    print(f"  REINFORCE best mean reward : {r_best:.3f}")
    print(f"  PPO       best mean reward : {p_best:.3f}")
    print(f"  Best models saved:")
    print(f"    -> models/pg/reinforce_best.pt")
    print(f"    -> models/pg/ppo_best.zip")
    print(f"  Plots + CSVs -> {RESULTS_DIR}")
    print("=" * 65)

    return r_df, p_df


if __name__ == "__main__":
    train()