"""
custom_env.py
─────────────────────────────────────────────────────────────────────────────
Custom Gymnasium Environment: SRH Education Platform
Mission: Train an RL agent to personalise Sexual & Reproductive Health (SRH)
         education for youth and people with disabilities.

Action Space  : Discrete(5)
Observation   : Box(6,) — float32, all values normalised [0, 1]
Reward        : Shaped (+10 to -10) based on learning outcomes
Max Steps     : 200 per episode
─────────────────────────────────────────────────────────────────────────────
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np


# ── Disability encoding ───────────────────────────────────────────────────
DISABILITY_MAP = {
    "none":      0.00,
    "visual":    0.33,
    "hearing":   0.66,
    "cognitive": 1.00,
}

DISABILITY_VALUES = list(DISABILITY_MAP.values())

# Actions appropriate for each disability (matched by nearest float)
ACCESSIBLE_ACTION_MAP = {
    0.00: [0, 1, 3, 4],
    0.33: [2, 4],
    0.66: [2, 4],
    1.00: [1, 2, 4],
}

ACTION_NAMES = {
    0: "Recommend Lesson",
    1: "Adjust Difficulty",
    2: "Provide Accessible Format",
    3: "Skip Topic",
    4: "Request Clarification",
}

# Observation indices
IDX_KNOWLEDGE   = 0
IDX_ENGAGEMENT  = 1
IDX_DISABILITY  = 2
IDX_PROGRESS    = 3
IDX_CONFUSION   = 4
IDX_PREV_ACTION = 5


def _nearest_disability_key(val: float, mapping: dict) -> float:
    """Return the nearest key in mapping to val — avoids float32 precision bugs."""
    return min(mapping.keys(), key=lambda k: abs(k - val))


def _disability_label(val: float) -> str:
    """Return the disability name string nearest to val."""
    return min(DISABILITY_MAP, key=lambda k: abs(DISABILITY_MAP[k] - val))


class SRHEducationEnv(gym.Env):
    """
    SRH Education Personalisation Environment
    ─────────────────────────────────────────
    The agent acts as an AI tutor selecting instructional strategies to
    maximise a learner's SRH knowledge and engagement while accommodating
    their disability needs.

    Observation Space — Box(6,) float32, all in [0, 1]:
    ┌──────────────────┬────────────────────────────────────────────────┐
    │ 0  knowledge     │ Retained SRH knowledge level                   │
    │ 1  engagement    │ Current engagement / motivation score           │
    │ 2  disability    │ Encoded disability (0=none…1=cognitive)         │
    │ 3  progress      │ Fraction of current topic completed             │
    │ 4  confusion     │ Whether learner shows confusion signals         │
    │ 5  prev_action   │ Last action taken (normalised to [0,1])         │
    └──────────────────┴────────────────────────────────────────────────┘

    Action Space — Discrete(5):
    ┌───┬──────────────────────────────────────────────────────────────┐
    │ 0 │ Recommend Lesson          — standard content delivery        │
    │ 1 │ Adjust Difficulty         — adapt content level              │
    │ 2 │ Provide Accessible Format — audio/visual/simplified text     │
    │ 3 │ Skip Topic                — move to next topic               │
    │ 4 │ Request Clarification     — prompt learner to ask questions  │
    └───┴──────────────────────────────────────────────────────────────┘

    Reward Structure:
    ┌──────┬─────────────────────────────────────────────────────────┐
    │ +10  │ Learner answers correctly (knowledge retained)          │
    │  +5  │ Engagement score increases after an action              │
    │  +3  │ Accessible format correctly matched to disability       │
    │  +1  │ Topic progress advances without confusion               │
    │   0  │ Neutral — no measurable outcome                         │
    │  -3  │ Redundant action (repeated completed lesson)            │
    │  -5  │ Confusion detected after action                         │
    │ -10  │ Learner dropout — episode terminates                    │
    └──────┴─────────────────────────────────────────────────────────┘
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, disability_type=None, max_steps=200):
        super().__init__()

        self.max_steps       = max_steps
        self.render_mode     = render_mode
        self.disability_type = disability_type

        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(
            low=np.zeros(6, dtype=np.float32),
            high=np.ones(6,  dtype=np.float32),
            dtype=np.float32,
        )

        self._state       = None
        self._step_count  = 0
        self._prev_action = 0
        self.window       = None
        self.clock        = None

    # ── Helpers ──────────────────────────────────────────────────────────

    def _get_obs(self):
        return self._state.copy()

    def _get_info(self):
        val   = float(self._state[IDX_DISABILITY])
        label = _disability_label(val)
        return {
            "step":         self._step_count,
            "knowledge":    round(float(self._state[IDX_KNOWLEDGE]),  3),
            "engagement":   round(float(self._state[IDX_ENGAGEMENT]), 3),
            "disability":   label,
            "progress":     round(float(self._state[IDX_PROGRESS]),   3),
            "confusion":    bool(self._state[IDX_CONFUSION] > 0.5),
            "action_taken": ACTION_NAMES[self._prev_action],
        }

    # ── Core API ─────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.disability_type is not None:
            disability_val = DISABILITY_MAP[self.disability_type]
        else:
            disability_val = float(self.np_random.choice(DISABILITY_VALUES))

        self._state = np.array([
            0.0,            # knowledge
            0.5,            # engagement
            disability_val, # disability
            0.0,            # progress
            0.0,            # confusion
            0.0,            # prev_action
        ], dtype=np.float32)

        self._step_count  = 0
        self._prev_action = 0
        return self._get_obs(), self._get_info()

    def step(self, action):
        assert self.action_space.contains(action), f"Invalid action: {action}"

        self._step_count  += 1
        self._prev_action  = int(action)
        reward             = 0.0
        terminated         = False
        rng                = self.np_random

        disability_val     = float(self._state[IDX_DISABILITY])
        d_key              = _nearest_disability_key(disability_val, ACCESSIBLE_ACTION_MAP)
        accessible_actions = ACCESSIBLE_ACTION_MAP[d_key]

        # ── Action dynamics ───────────────────────────────────────────────
        if action == 0:  # Recommend Lesson
            if self._state[IDX_PROGRESS] < 1.0:
                confusion_chance = 0.2 if self._state[IDX_ENGAGEMENT] > 0.5 else 0.4
                if rng.random() > confusion_chance:
                    gain = rng.uniform(0.05, 0.20)
                    self._state[IDX_KNOWLEDGE] = min(1.0, self._state[IDX_KNOWLEDGE] + gain)
                    self._state[IDX_PROGRESS]  = min(1.0, self._state[IDX_PROGRESS]  + 0.1)
                    self._state[IDX_CONFUSION] = 0.0
                    reward += 10.0 if rng.random() > 0.3 else 1.0
                else:
                    self._state[IDX_CONFUSION]  = 1.0
                    self._state[IDX_ENGAGEMENT] = max(0.0, self._state[IDX_ENGAGEMENT] - 0.1)
                    reward -= 5.0
            else:
                reward -= 3.0  # redundant

        elif action == 1:  # Adjust Difficulty
            if self._state[IDX_CONFUSION] > 0.5:
                self._state[IDX_CONFUSION]  = 0.0
                self._state[IDX_ENGAGEMENT] = min(1.0, self._state[IDX_ENGAGEMENT] + 0.1)
                reward += 5.0
            else:
                reward += 1.0

        elif action == 2:  # Provide Accessible Format
            if action in accessible_actions:
                self._state[IDX_CONFUSION]  = max(0.0, self._state[IDX_CONFUSION] - 0.5)
                self._state[IDX_ENGAGEMENT] = min(1.0, self._state[IDX_ENGAGEMENT] + 0.15)
                reward += 3.0
                if rng.random() > 0.4:
                    reward += 5.0
            else:
                reward += 0.0

        elif action == 3:  # Skip Topic
            if self._state[IDX_PROGRESS] > 0.5:
                self._state[IDX_PROGRESS]  = min(1.0, self._state[IDX_PROGRESS] + 0.2)
                self._state[IDX_CONFUSION] = 0.0
                reward += 1.0
            else:
                self._state[IDX_KNOWLEDGE] = max(0.0, self._state[IDX_KNOWLEDGE] - 0.05)
                reward -= 3.0

        elif action == 4:  # Request Clarification
            self._state[IDX_CONFUSION]  = max(0.0, self._state[IDX_CONFUSION] - 0.3)
            self._state[IDX_ENGAGEMENT] = min(1.0, self._state[IDX_ENGAGEMENT] + 0.05)
            reward += 1.0

        # ── Engagement decay ──────────────────────────────────────────────
        self._state[IDX_ENGAGEMENT] = max(
            0.0, self._state[IDX_ENGAGEMENT] - rng.uniform(0.0, 0.03)
        )

        # ── Dropout check ─────────────────────────────────────────────────
        dropout_prob = (
            (1.0 - self._state[IDX_ENGAGEMENT]) * 0.15
            + self._state[IDX_CONFUSION]         * 0.10
        )
        if rng.random() < dropout_prob:
            reward    -= 10.0
            terminated = True

        # ── Success check ─────────────────────────────────────────────────
        if self._state[IDX_PROGRESS] >= 1.0 and self._state[IDX_KNOWLEDGE] >= 0.7:
            reward    += 10.0
            terminated = True

        # ── Timeout ───────────────────────────────────────────────────────
        truncated = self._step_count >= self.max_steps

        self._state[IDX_PREV_ACTION] = int(action) / 4.0

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    # ── Rendering ─────────────────────────────────────────────────────────

    def render(self):
        if self.render_mode == "human":
            self._render_pygame()
        elif self.render_mode == "rgb_array":
            self._render_pygame()
            import pygame
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.window)), axes=(1, 0, 2)
            )

    def _render_pygame(self):
        import pygame
        W, H = 620, 440

        if self.window is None:
            pygame.init()
            pygame.display.set_caption("SRH Education RL — Agent Dashboard")
            self.window = pygame.display.set_mode((W, H))
            self.clock  = pygame.time.Clock()

        canvas = pygame.Surface((W, H))
        canvas.fill((15, 20, 40))

        f_title = pygame.font.SysFont("Arial", 19, bold=True)
        f_label = pygame.font.SysFont("Arial", 14)
        f_small = pygame.font.SysFont("Arial", 12)

        # Title
        t = f_title.render("SRH Education Platform — RL Agent Dashboard", True, (100, 200, 255))
        canvas.blit(t, (W // 2 - t.get_width() // 2, 12))

        # Bars
        labels  = ["Knowledge", "Engagement", "Progress", "Confusion"]
        indices = [IDX_KNOWLEDGE, IDX_ENGAGEMENT, IDX_PROGRESS, IDX_CONFUSION]
        colors  = [(80, 200, 120), (80, 160, 240), (255, 200, 60), (240, 80, 80)]
        bar_y, bar_max, gap = 55, 160, 130

        for i, (lbl, idx, col) in enumerate(zip(labels, indices, colors)):
            val   = float(self._state[idx])
            bar_h = int(val * bar_max)
            x     = 30 + i * gap

            pygame.draw.rect(canvas, (35, 45, 65),
                             pygame.Rect(x, bar_y, 100, bar_max), border_radius=6)
            if bar_h > 0:
                pygame.draw.rect(canvas, col,
                                 pygame.Rect(x, bar_y + bar_max - bar_h, 100, bar_h),
                                 border_radius=6)

            canvas.blit(f_label.render(lbl, True, (200, 210, 230)),
                        (x + 4,  bar_y + bar_max + 5))
            canvas.blit(f_small.render(f"{val:.2f}", True, col),
                        (x + 36, bar_y + bar_max - bar_h - 17))

        # Info panel
        disability_label = _disability_label(float(self._state[IDX_DISABILITY]))
        panel_y = 255
        lines = [
            (f"Disability Type : {disability_label.upper()}",   (255, 220, 100)),
            (f"Last Action     : {ACTION_NAMES[self._prev_action]}", (180, 230, 180)),
            (f"Step            : {self._step_count} / {self.max_steps}", (160, 170, 200)),
            (f"Knowledge       : {self._state[IDX_KNOWLEDGE]:.3f}", (80, 200, 120)),
            (f"Engagement      : {self._state[IDX_ENGAGEMENT]:.3f}", (80, 160, 240)),
            (f"Progress        : {self._state[IDX_PROGRESS]:.3f}",   (255, 200, 60)),
        ]
        for i, (text, col) in enumerate(lines):
            canvas.blit(f_label.render(text, True, col), (30, panel_y + i * 22))

        # Divider
        pygame.draw.line(canvas, (45, 55, 80), (20, 395), (W - 20, 395), 1)

        # Footer
        canvas.blit(
            f_small.render(
                "Mission: Personalised SRH Education for Youth & People with Disabilities",
                True, (80, 95, 125)
            ), (20, 408)
        )

        self.window.blit(canvas, (0, 0))
        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.window is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self.window = None

    # ── JSON serialisation (for API / frontend integration) ───────────────

    def to_json(self):
        """Serialise current state to JSON-compatible dict for API use."""
        import json
        state_dict = {
            "step":          self._step_count,
            "max_steps":     self.max_steps,
            "knowledge":     round(float(self._state[IDX_KNOWLEDGE]),  4),
            "engagement":    round(float(self._state[IDX_ENGAGEMENT]), 4),
            "disability":    _disability_label(float(self._state[IDX_DISABILITY])),
            "progress":      round(float(self._state[IDX_PROGRESS]),   4),
            "confusion":     bool(self._state[IDX_CONFUSION] > 0.5),
            "last_action":   ACTION_NAMES[self._prev_action],
            "action_space":  list(ACTION_NAMES.values()),
            "observation":   [round(float(x), 4) for x in self._state],
        }
        return json.dumps(state_dict, indent=2)