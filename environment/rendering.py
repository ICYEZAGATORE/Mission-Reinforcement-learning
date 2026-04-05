"""
rendering.py
─────────────────────────────────────────────────────────────────────────────
Visualisation of the SRH Education Environment.

Modes:
  --mode random   → Random agent demo (no model, no training)
  --mode diagram  → Static architectural diagram of agent + environment

Run:
    python environment/rendering.py --mode random
    python environment/rendering.py --mode diagram
─────────────────────────────────────────────────────────────────────────────
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pygame
import numpy as np
from environment.custom_env import SRHEducationEnv, ACTION_NAMES


# ─────────────────────────────────────────────────────────────────────────
# RANDOM AGENT DEMO
# ─────────────────────────────────────────────────────────────────────────

def run_random_agent(num_steps: int = 200, fps: int = 6):
    """
    Runs the SRH environment with a purely RANDOM agent.
    No model is used — actions are sampled uniformly.
    Demonstrates all environment components visually.
    """
    env          = SRHEducationEnv(render_mode="human", max_steps=num_steps)
    obs, info    = env.reset(seed=42)
    total_reward = 0.0
    step         = 0
    episode      = 1

    pygame.init()
    clock = pygame.time.Clock()

    print("=" * 65)
    print("  SRH Education Environment — Random Agent Visualisation")
    print("  (No model — purely random actions, no training)")
    print("=" * 65)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        if not running:
            break

        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step         += 1

        print(
            f"[Ep {episode:02d} | Step {step:03d}] "
            f"Action: {ACTION_NAMES[action]:<28} "
            f"Reward: {reward:+6.1f}  Total: {total_reward:+7.1f}  "
            f"Knowledge: {info['knowledge']:.2f}  "
            f"Engagement: {info['engagement']:.2f}  "
            f"Confusion: {info['confusion']}"
        )

        env.render()
        clock.tick(fps)

        if terminated or truncated:
            status = "SUCCESS" if (terminated and reward >= 0) else "DROPOUT/TIMEOUT"
            print(f"\n  Episode {episode} ended [{status}] "
                  f"| Total reward: {total_reward:+.1f}\n")
            episode      += 1
            total_reward  = 0.0
            step          = 0
            obs, info     = env.reset()

    env.close()
    print("\nVisualization closed.")


# ─────────────────────────────────────────────────────────────────────────
# ENVIRONMENT DIAGRAM
# ─────────────────────────────────────────────────────────────────────────

def draw_environment_diagram():
    """
    Draws a static diagram showing:
      - RL Agent box
      - Environment (SRH Learner) box
      - Action arrow, Observation arrow, Reward arc
      - Action list, Observation features, Reward structure legend
    Saves as environment_diagram.png
    """
    W, H = 820, 520
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("SRH RL — Agent & Environment Diagram")

    BG     = (15, 20, 40)
    BLUE   = (55, 125, 220)
    GREEN  = (55, 185, 110)
    ORANGE = (255, 155, 45)
    YELLOW = (255, 215, 55)
    RED    = (230, 75,  75)
    WHITE  = (215, 225, 250)
    GREY   = (90,  105, 135)
    DARK   = (25,  35,  60)

    f_big   = pygame.font.SysFont("Arial", 17, bold=True)
    f_med   = pygame.font.SysFont("Arial", 13)
    f_small = pygame.font.SysFont("Arial", 11)

    screen.fill(BG)

    # Title
    title = f_big.render(
        "SRH Education RL — Agent in Simulated Environment", True, WHITE
    )
    screen.blit(title, (W // 2 - title.get_width() // 2, 18))

    # Agent box
    agent_rect = pygame.Rect(60, 160, 200, 130)
    pygame.draw.rect(screen, BLUE, agent_rect, border_radius=12)
    pygame.draw.rect(screen, WHITE, agent_rect, 2, border_radius=12)
    screen.blit(f_big.render("RL AGENT", True, WHITE),          (105, 183))
    screen.blit(f_med.render("DQN  |  REINFORCE", True, WHITE), (80,  208))
    screen.blit(f_med.render("PPO  |  A2C",       True, WHITE), (100, 228))

    # Environment box
    env_rect = pygame.Rect(555, 160, 215, 130)
    pygame.draw.rect(screen, GREEN, env_rect, border_radius=12)
    pygame.draw.rect(screen, WHITE, env_rect, 2, border_radius=12)
    screen.blit(f_big.render("ENVIRONMENT", True, DARK),   (573, 183))
    screen.blit(f_med.render("SRH Learner",   True, DARK), (600, 208))
    screen.blit(f_med.render("(Gymnasium)",   True, DARK), (600, 228))

    # Action arrow (Agent → Environment)
    pygame.draw.line(screen, ORANGE, (260, 200), (555, 200), 3)
    pygame.draw.polygon(screen, ORANGE, [(555, 193), (570, 200), (555, 207)])
    screen.blit(f_med.render("Action  (Discrete 5)", True, ORANGE), (340, 178))

    action_items = [
        "0: Recommend Lesson",
        "1: Adjust Difficulty",
        "2: Accessible Format",
        "3: Skip Topic",
        "4: Request Clarification",
    ]
    for i, item in enumerate(action_items):
        screen.blit(f_small.render(item, True, ORANGE), (305, 320 + i * 16))

    # Observation arrow (Environment → Agent)
    pygame.draw.line(screen, YELLOW, (555, 250), (260, 250), 3)
    pygame.draw.polygon(screen, YELLOW, [(260, 243), (245, 250), (260, 257)])
    screen.blit(f_med.render("Observation  (Box 6D)", True, YELLOW), (330, 258))

    obs_items = [
        "knowledge | engagement",
        "disability | progress",
        "confusion | prev_action",
    ]
    for i, item in enumerate(obs_items):
        screen.blit(f_small.render(item, True, YELLOW), (330, 278 + i * 14))

    # Reward arc
    pygame.draw.arc(screen, RED, pygame.Rect(240, 320, 350, 80), 0, 3.14, 3)
    screen.blit(f_med.render("Reward  (+10 to -10)", True, RED), (330, 405))

    # Legend — reward structure
    legend = pygame.Rect(28, 355, 230, 130)
    pygame.draw.rect(screen, DARK, legend, border_radius=8)
    pygame.draw.rect(screen, GREY, legend, 1, border_radius=8)
    screen.blit(f_med.render("Reward Structure", True, WHITE), (50, 362))
    rewards = [
        ("+10  correct answer",    GREEN),
        ("+5   engagement ↑",      GREEN),
        ("+3   accessibility match", GREEN),
        ("-5   confusion detected", RED),
        ("-10  learner dropout",   RED),
    ]
    for i, (text, col) in enumerate(rewards):
        screen.blit(f_small.render(text, True, col), (38, 382 + i * 17))

    # Legend — observation features
    obs_legend = pygame.Rect(555, 315, 230, 130)
    pygame.draw.rect(screen, DARK, obs_legend, border_radius=8)
    pygame.draw.rect(screen, GREY, obs_legend, 1, border_radius=8)
    screen.blit(f_med.render("Observation Features", True, WHITE), (568, 322))
    obs_feat = [
        "knowledge    [0.0 – 1.0]",
        "engagement   [0.0 – 1.0]",
        "disability   [0, 0.33, 0.66, 1]",
        "progress     [0.0 – 1.0]",
        "confusion    [0 or 1]",
        "prev_action  [0.0 – 1.0]",
    ]
    for i, text in enumerate(obs_feat):
        screen.blit(f_small.render(text, True, YELLOW), (563, 342 + i * 16))

    # Footer
    screen.blit(
        f_small.render(
            "Capstone: AI-Powered SRH Education for Youth & People with Disabilities",
            True, GREY
        ), (20, H - 22)
    )

    pygame.display.update()

    # Save PNG
    out = os.path.join(os.path.dirname(__file__), "..", "environment_diagram.png")
    pygame.image.save(screen, out)
    print(f"Diagram saved → {os.path.abspath(out)}")

    running = True
    while running:
        for event in pygame.event.get():
            if event.type in (pygame.QUIT,):
                running = False
            if event.type == pygame.KEYDOWN:
                running = False
    pygame.quit()


# ─────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",  choices=["random", "diagram"], default="random")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--fps",   type=int, default=6)
    args = parser.parse_args()

    if args.mode == "random":
        run_random_agent(num_steps=args.steps, fps=args.fps)
    else:
        draw_environment_diagram()