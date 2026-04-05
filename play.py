"""
play.py  (also works as main.py)
─────────────────────────────────────────────────────────────────────────────
Entry point for running the best-performing RL agent in the SRH Education
Environment. This is what the rubric refers to as "play.py".

Usage:
    python play.py --algo dqn          # Run best DQN agent
    python play.py --algo ppo          # Run best PPO agent
    python play.py --algo a2c          # Run best A2C agent
    python play.py --algo reinforce    # Run best REINFORCE agent
    python play.py --compare           # Print comparison table of all models
    python play.py --api               # Print JSON state (API demo)

Training (run these separately):
    python training/dqn_training.py
    python training/pg_training.py
─────────────────────────────────────────────────────────────────────────────
"""

import argparse
import os
import sys
import numpy as np

# ── Path setup — works regardless of where you call from ─────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from environment.custom_env import SRHEducationEnv, ACTION_NAMES


# ─────────────────────────────────────────────────────────────────────────
# Run SB3 model (DQN / PPO / A2C)
# ─────────────────────────────────────────────────────────────────────────

def run_sb3(model_path: str, algo: str, n_episodes: int = 5):
    from stable_baselines3 import DQN, PPO, A2C
    import pygame

    CLS_MAP = {"dqn": DQN, "ppo": PPO, "a2c": A2C}
    model   = CLS_MAP[algo].load(model_path)

    env   = SRHEducationEnv(render_mode="human", max_steps=200)
    pygame.init()
    clock = pygame.time.Clock()

    print(f"\n  Running {algo.upper()} agent | {n_episodes} episodes")
    print("  Press ESC or close the window to stop.\n")

    for ep in range(n_episodes):
        obs, info = env.reset()
        total, step, done = 0.0, 0, False
        print(f"  ── Episode {ep+1} ──")

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close(); pygame.quit(); return
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    env.close(); pygame.quit(); return

            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            total += reward
            step  += 1
            done   = terminated or truncated

            print(
                f"  Step {step:03d} | {ACTION_NAMES[int(action)]:<28} "
                f"| Reward: {reward:+6.1f} | Total: {total:+7.1f} "
                f"| Knowledge: {info['knowledge']:.3f} "
                f"| Engagement: {info['engagement']:.3f} "
                f"| Confusion: {info['confusion']}"
            )
            env.render()
            clock.tick(6)

        status = "SUCCESS" if (terminated and reward >= 0) else "DROPOUT/TIMEOUT"
        print(f"\n  Episode {ep+1} [{status}] | Total reward: {total:+.1f}\n")

    env.close()
    pygame.quit()


# ─────────────────────────────────────────────────────────────────────────
# Run REINFORCE model
# ─────────────────────────────────────────────────────────────────────────

def run_reinforce(model_path: str, n_episodes: int = 5):
    import torch
    import pygame
    from training.pg_training import PolicyNet

    env        = SRHEducationEnv(render_mode="human", max_steps=200)
    obs_dim    = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy = PolicyNet(obs_dim, action_dim)
    policy.load_state_dict(torch.load(model_path, map_location="cpu"))
    policy.eval()

    pygame.init()
    clock = pygame.time.Clock()

    print(f"\n  Running REINFORCE agent | {n_episodes} episodes\n")

    for ep in range(n_episodes):
        obs, _ = env.reset()
        total, step, done = 0.0, 0, False
        print(f"  ── Episode {ep+1} ──")

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close(); pygame.quit(); return
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    env.close(); pygame.quit(); return

            with torch.no_grad():
                probs  = policy(torch.FloatTensor(obs).unsqueeze(0))
                action = probs.argmax().item()

            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            step  += 1
            done   = terminated or truncated

            print(
                f"  Step {step:03d} | {ACTION_NAMES[action]:<28} "
                f"| Reward: {reward:+6.1f} | Total: {total:+7.1f} "
                f"| Knowledge: {info['knowledge']:.3f} "
                f"| Engagement: {info['engagement']:.3f}"
            )
            env.render()
            clock.tick(6)

        print(f"\n  Episode {ep+1} done | Total: {total:+.1f}\n")

    env.close()
    pygame.quit()


# ─────────────────────────────────────────────────────────────────────────
# Compare all trained models
# ─────────────────────────────────────────────────────────────────────────

def compare_all():
    from stable_baselines3 import DQN, PPO, A2C
    from stable_baselines3.common.evaluation import evaluate_policy
    import pandas as pd

    print("\n  Comparing all trained models (30 eval episodes each)...\n")
    rows = []

    sb3_models = [
        ("DQN",  os.path.join(BASE_DIR, "models", "dqn", "dqn_best"), DQN),
        ("PPO",  os.path.join(BASE_DIR, "models", "pg",  "ppo_best"), PPO),
        ("A2C",  os.path.join(BASE_DIR, "models", "pg",  "a2c_best"), A2C),
    ]

    for name, path, cls in sb3_models:
        zip_path = path + ".zip"
        if not os.path.exists(zip_path):
            print(f"  [{name}] Not found at {zip_path} — train first.")
            continue
        model    = cls.load(path)
        eval_env = SRHEducationEnv(max_steps=200)
        mean_r, std_r = evaluate_policy(model, eval_env, n_eval_episodes=30, deterministic=True)
        eval_env.close()
        print(f"  [{name}] Mean: {mean_r:.3f} ± {std_r:.3f}")
        rows.append({"Algorithm": name, "Mean Reward": round(mean_r, 3), "Std": round(std_r, 3)})

    # REINFORCE
    reinforce_path = os.path.join(BASE_DIR, "models", "pg", "reinforce_best.pt")
    if os.path.exists(reinforce_path):
        import torch
        from training.pg_training import PolicyNet
        eval_env   = SRHEducationEnv(max_steps=200)
        obs_dim    = eval_env.observation_space.shape[0]
        action_dim = eval_env.action_space.n
        policy     = PolicyNet(obs_dim, action_dim)
        policy.load_state_dict(torch.load(reinforce_path, map_location="cpu"))
        policy.eval()
        ep_rs = []
        with torch.no_grad():
            for _ in range(30):
                obs, _ = eval_env.reset()
                total, done = 0.0, False
                while not done:
                    a   = policy(torch.FloatTensor(obs).unsqueeze(0)).argmax().item()
                    obs, r, t, tr, _ = eval_env.step(a)
                    total += r
                    done   = t or tr
                ep_rs.append(total)
        eval_env.close()
        mean_r, std_r = np.mean(ep_rs), np.std(ep_rs)
        print(f"  [REINFORCE] Mean: {mean_r:.3f} ± {std_r:.3f}")
        rows.append({"Algorithm": "REINFORCE", "Mean Reward": round(float(mean_r), 3),
                     "Std": round(float(std_r), 3)})

    if rows:
        df = pd.DataFrame(rows).sort_values("Mean Reward", ascending=False)
        print("\n  ── Final Model Comparison ──")
        print(df.to_string(index=False))
        out = os.path.join(BASE_DIR, "models", "comparison_results.csv")
        df.to_csv(out, index=False)
        print(f"\n  Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────
# JSON API demo — shows how state can be served to a frontend
# ─────────────────────────────────────────────────────────────────────────

def api_demo():
    env    = SRHEducationEnv(max_steps=200)
    obs, _ = env.reset(seed=0)
    action = env.action_space.sample()
    env.step(action)

    print("\n  ── JSON State (API Output) ──")
    print("  This is how the environment state can be serialised")
    print("  and served to a web/mobile frontend via an API endpoint.\n")
    print(env.to_json())
    env.close()


# ─────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SRH Education RL — Play / Evaluate Best Agent"
    )
    parser.add_argument(
        "--algo", choices=["dqn", "ppo", "a2c", "reinforce"], default="dqn",
        help="Algorithm to run"
    )
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--compare",  action="store_true", help="Compare all models")
    parser.add_argument("--api",      action="store_true", help="JSON API state demo")
    args = parser.parse_args()

    print("=" * 65)
    print("  SRH Education Platform — Mission-Based RL")
    print("  AI Education for Youth & People with Disabilities")
    print("=" * 65)

    if args.compare:
        compare_all()
        return

    if args.api:
        api_demo()
        return

    algo = args.algo.lower()

    if algo in ("dqn", "ppo", "a2c"):
        folder = "dqn" if algo == "dqn" else "pg"
        path   = os.path.join(BASE_DIR, "models", folder, f"{algo}_best")
        if not os.path.exists(path + ".zip"):
            print(f"\n  Model not found: {path}.zip")
            print("  Train first:  python training/dqn_training.py")
            print("                python training/pg_training.py")
            return
        run_sb3(path, algo, n_episodes=args.episodes)

    elif algo == "reinforce":
        path = os.path.join(BASE_DIR, "models", "pg", "reinforce_best.pt")
        if not os.path.exists(path):
            print(f"\n  Model not found: {path}")
            print("  Train first:  python training/pg_training.py")
            return
        run_reinforce(path, n_episodes=args.episodes)


if __name__ == "__main__":
    main()