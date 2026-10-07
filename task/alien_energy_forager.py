"""
Alien Energy Forager
====================
A two-option restless foraging task (hidden-state reversal task) in Pygame.

Two identical crystals sit on an alien planet. At any moment exactly one of
them is secretly "active". Every TRIAL_INTERVAL_MS the alien's extractor ray
fires at whichever crystal it is currently aimed at:
    - active crystal   -> reward with probability P_REWARD
    - inactive crystal -> never rewarded
Each firing at the active crystal depletes it with probability P_SWITCH, and the
other crystal silently becomes active. Firing at the inactive crystal changes
nothing, so once a crystal is depleted it stays empty until the player goes to
forage the other one. The player can re-aim at any time with LEFT / RIGHT.

Events (trial, reward, fail, switch, system_flip) are written as they happen to
task_events.csv in the output folder: the folder given by --out-dir (the recorder
passes the session folder), or, when run standalone, a new
task/data/alien_energy_forager_YYYYMMDD_HHMMSS/ folder.

Controls: SPACE = start, LEFT/RIGHT = aim, ESC = quit (data is saved after every event).
Requires: pygame. numpy is used for sound generation if installed; otherwise a
standard-library fallback is used.

facepipe: emit() prints one "FACEPIPE_EVENT <name> <json>" line on stdout per event
(ready, task_start, every logged event, task_end, quit), which
scripts/recording_alien_forager.py reads to time the webcam recording.
The game still runs on its own.
"""

import argparse
import csv
import json
import math
import os
import random
import sys
from array import array
from datetime import datetime

import pygame

try:
    import numpy as np
except ImportError:
    np = None

# ---------------------------------------------------------------------------
# Task parameters
# ---------------------------------------------------------------------------
N_TRIALS = 300
P_REWARD = 0.9            # reward probability when aiming at the active crystal
P_SWITCH = 0.3            # probability a firing at the active crystal depletes it
TRIAL_INTERVAL_MS = 1000  # time between automatic firings
RANDOM_SEED = None        # set to an int for a reproducible reward/switch sequence

# ---------------------------------------------------------------------------
# Display / timing
# ---------------------------------------------------------------------------
SCREEN_W, SCREEN_H = 800, 600
PIXEL = 4                              # size of one "art pixel" on screen
CANVAS_W, CANVAS_H = SCREEN_W // PIXEL, SCREEN_H // PIXEL
FPS = 60
N_STARS = 90
FLASH_MS = 200                         # crystal feedback flash duration
FIRE_PULSE_MS = 120                    # brief solid beam when the ray fires
END_DELAY_MS = 800                     # pause after last trial before end screen
BOB_PERIOD_S = 1.6                     # alien idle bob cycle
BOB_AMPLITUDE = 1.5                    # in art pixels

# Energy meter is full when this many rewards have been collected
METER_CAPACITY = N_TRIALS * P_REWARD

# ---------------------------------------------------------------------------
# Data logging
# ---------------------------------------------------------------------------
TASK_NAME = "alien_energy_forager"
DATA_DIR = "data"                     # standalone runs: data/alien_energy_forager_YYYYMMDD_HHMMSS/
DATA_FILE = "task_events.csv"
# One row per event. frame_idx is left empty here; the recorder fills it in after
# the session (nearest video frame). Events:
#   trial        a trial happens at `side` (automatic firing, every TRIAL_INTERVAL_MS)
#   reward/fail  that trial's outcome (same timestamp as the trial)
#   switch       the participant leaves the current site; side = the site they go to
#                (the player re-aims; instant)
#   system_flip  hidden: the active site flips; side = the newly active site
CSV_COLUMNS = ["timestamp", "frame_idx", "event", "side"]

INSTRUCTIONS = [
    "Help the alien harvest energy from the crystals.",
    "Aim the ray with the LEFT and RIGHT arrow keys.",
    "The ray fires by itself. Only one crystal holds",
    "energy at a time, and that can change.",
]

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
SKY = (8, 6, 24)
STAR_WHITE = (235, 235, 255)
STAR_YELLOW = (255, 230, 140)
GROUND = (92, 78, 112)
GROUND_RIM = (128, 112, 150)
CRATER = (72, 60, 90)
SHADOW = (58, 48, 74)

ALIEN_BODY = (232, 92, 124)
ALIEN_LIGHT = (255, 160, 180)
ALIEN_DARK = (140, 36, 66)
ANTENNA = (255, 240, 120)
BLACK = (10, 10, 16)
WHITE = (255, 255, 255)

CRYSTAL_NORMAL = {"light": (90, 220, 240), "dark": (40, 150, 200),
                  "highlight": (200, 250, 255), "outline": (20, 60, 110)}
CRYSTAL_BRIGHT = {"light": (255, 255, 255), "dark": (180, 250, 255),
                  "highlight": (255, 255, 255), "outline": (90, 200, 240)}
CRYSTAL_DIM = {"light": (50, 90, 110), "dark": (30, 60, 80),
               "highlight": (80, 110, 120), "outline": (15, 35, 60)}

RAY_COLOR = (200, 255, 80)
RAY_BRIGHT = (250, 255, 190)

METER_FRAME = (200, 200, 220)
METER_BG = (25, 22, 45)
METER_FILL = (200, 255, 80)
METER_FILL_LIGHT = (240, 255, 180)
METER_SEG = (120, 170, 50)

TEXT_COLOR = (230, 230, 255)
TITLE_COLOR = (255, 230, 120)
PANEL_BG = (10, 8, 30, 215)
PANEL_BORDER = (128, 112, 150)

# ---------------------------------------------------------------------------
# Scene layout (in canvas / art-pixel coordinates)
# ---------------------------------------------------------------------------
GROUND_RECT = (-60, 95, 320, 160)      # big ellipse -> curved planet horizon
CRATERS = [(22, 128, 16, 4), (140, 134, 18, 5), (84, 136, 12, 3),
           (168, 112, 10, 3), (8, 112, 9, 3)]
ALIEN_X, ALIEN_BASE_Y = 100, 100
CRYSTAL_X = {"left": 48, "right": 152}
CRYSTAL_BASE_Y = 114
CRYSTAL_H, CRYSTAL_HALF_W = 20, 6
METER_RECT = (184, 40, 9, 80)          # x, y, w, h


def other(side):
    return "right" if side == "left" else "left"


def emit(event, **fields):
    """One machine-readable line on stdout for the facepipe recorder; harmless otherwise."""
    fields = {"t": datetime.now().isoformat(timespec="microseconds"), **fields}
    try:
        print("FACEPIPE_EVENT", event, json.dumps(fields), flush=True)
    except (OSError, ValueError):   # recorder gone / stdout closed
        pass


# ---------------------------------------------------------------------------
# Sound synthesis
# ---------------------------------------------------------------------------
def _reward_samples(sr):
    """Bright ascending three-note chime, ~220 ms."""
    notes = [(0.00, 988.0), (0.06, 1319.0), (0.12, 1760.0)]
    n = int(0.22 * sr)
    out = []
    for i in range(n):
        t = i / sr
        s = 0.0
        for t0, f in notes:
            if t >= t0:
                dt = t - t0
                env = math.exp(-dt * 22.0)
                s += env * (math.sin(2 * math.pi * f * dt)
                            + 0.25 * math.sin(4 * math.pi * f * dt))
        fade = min(1.0, (n - i) / (0.01 * sr))   # avoid a click at the end
        out.append(0.28 * s * fade)
    return out


def _failure_samples(sr):
    """Dull low buzz, ~200 ms, with a slight downward pitch drift."""
    n = int(0.20 * sr)
    out = []
    phase = 0.0
    for i in range(n):
        t = i / sr
        f = 95.0 - 25.0 * (t / 0.20)
        phase += 2 * math.pi * f / sr
        s = (math.sin(phase) + 0.5 * math.sin(2 * phase)
             + 0.33 * math.sin(3 * phase) + 0.2 * math.sin(4 * phase))
        attack = min(1.0, t / 0.005)
        decay = 1.0 - i / n
        out.append(0.22 * s * attack * decay)
    return out


def _to_sound(samples, channels):
    ints = [int(max(-1.0, min(1.0, s)) * 32767) for s in samples]
    if np is not None:
        arr = np.array(ints, dtype=np.int16)
        if channels > 1:
            arr = np.ascontiguousarray(np.repeat(arr[:, None], channels, axis=1))
        return pygame.sndarray.make_sound(arr)
    buf = array("h", [v for v in ints for _ in range(channels)])
    return pygame.mixer.Sound(buffer=buf.tobytes())


def make_sounds():
    """Return (reward_sound, failure_sound), or (None, None) if audio is unavailable."""
    init = pygame.mixer.get_init()
    if init is None:
        return None, None
    sr, _fmt, channels = init
    try:
        return (_to_sound(_reward_samples(sr), channels),
                _to_sound(_failure_samples(sr), channels))
    except (pygame.error, ValueError) as exc:
        print(f"Sound disabled: {exc}")
        return None, None


# ---------------------------------------------------------------------------
# Game
# ---------------------------------------------------------------------------
class Game:
    def __init__(self, out_dir=None):
        pygame.mixer.pre_init(44100, -16, 1, 512)
        pygame.init()
        pygame.display.set_caption("Alien Energy Forager")
        self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        self.canvas = pygame.Surface((CANVAS_W, CANVAS_H))
        self.clock = pygame.time.Clock()
        self.fonts = {}
        self.rng = random.Random(RANDOM_SEED)
        self.snd_reward, self.snd_failure = make_sounds()

        star_rng = random.Random()
        self.stars = []
        for _ in range(N_STARS):
            x = star_rng.randrange(CANVAS_W)
            y = star_rng.randrange(0, 115)
            color = STAR_YELLOW if star_rng.random() < 0.25 else STAR_WHITE
            big = star_rng.random() < 0.08
            self.stars.append((x, y, color, big))

        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        if out_dir is None:
            out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), DATA_DIR,
                                   "{}_{}".format(TASK_NAME, self.session_id))
        os.makedirs(out_dir, exist_ok=True)
        self.data_path = os.path.join(out_dir, DATA_FILE)
        with open(self.data_path, "w", newline="") as f:
            csv.writer(f).writerow(CSV_COLUMNS)

        self.state = "start"
        self.aim = self.rng.choice(["left", "right"])
        self.active = self.rng.choice(["left", "right"])
        self.trial = 0
        self.energy = 0
        self.shown_energy = 0.0
        self.next_fire_ms = 0
        self.last_fire_ms = 0
        self.aim_since_ms = 0
        self.flash_side = None
        self.flash_kind = None
        self.flash_start_ms = -10_000
        emit("ready", session_id=self.session_id, data_path=self.data_path,
             n_trials=N_TRIALS, p_reward=P_REWARD, p_switch=P_SWITCH,
             trial_interval_ms=TRIAL_INTERVAL_MS, random_seed=RANDOM_SEED, fps=FPS,
             start_side=self.aim, start_active=self.active)

    # ----- task logic ------------------------------------------------------
    def start_task(self, now):
        self.state = "task"
        self.last_fire_ms = now
        self.aim_since_ms = now
        self.next_fire_ms = now + TRIAL_INTERVAL_MS
        emit("task_start", session_id=self.session_id, side=self.aim)

    def set_aim(self, side, now):
        if side != self.aim:
            self.aim = side
            self.aim_since_ms = now
            self.log_event("switch", side, datetime.now())

    def fire(self, now):
        self.trial += 1
        choice = self.aim
        active = self.active
        success = choice == active and self.rng.random() < P_REWARD
        # Only mining the active crystal can deplete it; the inactive one never
        # reactivates on its own, so the player must go to the other crystal.
        switched = choice == active and self.rng.random() < P_SWITCH

        t = datetime.now()
        self.log_event("trial", choice, t)
        self.log_event("reward" if success else "fail", choice, t)
        if switched:
            self.log_event("system_flip", other(active), t)

        if switched:
            self.active = other(active)
        if success:
            self.energy += 1
        self.flash_side = choice
        self.flash_kind = "success" if success else "failure"
        self.flash_start_ms = now
        self.last_fire_ms = now

        sound = self.snd_reward if success else self.snd_failure
        if sound is not None:
            sound.play()

    def log_event(self, event, side, t):
        """Append one event row (saved immediately) and tell the recorder on stdout."""
        stamp = t.isoformat(timespec="microseconds")
        with open(self.data_path, "a", newline="") as f:
            csv.writer(f).writerow([stamp, "", event, side])
        emit(event, side=side, t=stamp)

    # ----- main loop -------------------------------------------------------
    def run(self):
        while True:
            dt = self.clock.tick(FPS) / 1000.0
            now = pygame.time.get_ticks()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.quit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.quit()
                    elif self.state == "start" and event.key == pygame.K_SPACE:
                        self.start_task(now)
                    elif self.state == "task" and event.key == pygame.K_LEFT:
                        self.set_aim("left", now)
                    elif self.state == "task" and event.key == pygame.K_RIGHT:
                        self.set_aim("right", now)

            if self.state == "task":
                # Scheduled on a fixed grid so firings don't drift
                while self.trial < N_TRIALS and now >= self.next_fire_ms:
                    self.fire(now)
                    self.next_fire_ms += TRIAL_INTERVAL_MS
                if self.trial >= N_TRIALS and now - self.last_fire_ms >= END_DELAY_MS:
                    self.state = "end"
                    emit("task_end", trials=self.trial, n_rewards=self.energy)

            self.shown_energy += (self.energy - self.shown_energy) * min(1.0, dt * 8)
            self.draw(now)

    def quit(self):
        emit("quit", state=self.state, trials=self.trial, n_rewards=self.energy)
        pygame.quit()
        sys.exit()

    # ----- drawing ---------------------------------------------------------
    def draw(self, now):
        c = self.canvas
        c.fill(SKY)
        self.draw_stars(c)
        self.draw_ground(c)
        tip = self.draw_alien(c, now / 1000.0)
        if self.state == "task":
            self.draw_ray(c, tip, now)
        for side in ("left", "right"):
            self.draw_crystal(c, side, now)
        self.draw_meter(c)

        pygame.transform.scale(c, (SCREEN_W, SCREEN_H), self.screen)
        mx, my, mw, _ = METER_RECT
        self.text("ENERGY", 18, TEXT_COLOR, ((mx + mw / 2) * PIXEL, my * PIXEL - 16))

        if self.state == "start":
            self.draw_start_screen(now)
        elif self.state == "end":
            self.draw_end_screen()
        pygame.display.flip()

    def draw_stars(self, c):
        for x, y, color, big in self.stars:
            c.set_at((x, y), color)
            if big:
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    c.set_at((x + dx, y + dy), color)

    def draw_ground(self, c):
        pygame.draw.ellipse(c, GROUND, GROUND_RECT)
        pygame.draw.ellipse(c, GROUND_RIM, GROUND_RECT, 1)
        for rect in CRATERS:
            pygame.draw.ellipse(c, CRATER, rect)

    def draw_alien(self, c, t):
        """Draw the alien; returns the antenna tip (ray origin)."""
        bob = round(BOB_AMPLITUDE * math.sin(2 * math.pi * t / BOB_PERIOD_S))
        x, base = ALIEN_X, ALIEN_BASE_Y
        pygame.draw.ellipse(c, SHADOW, (x - 12, base - 2, 24, 5))
        pygame.draw.rect(c, ALIEN_DARK, (x - 7, base - 3, 5, 3))
        pygame.draw.rect(c, ALIEN_DARK, (x + 3, base - 3, 5, 3))

        top = base - 22 + bob
        pygame.draw.line(c, ALIEN_DARK, (x, top), (x, top - 5))
        pygame.draw.rect(c, ANTENNA, (x - 1, top - 7, 3, 3))
        pygame.draw.rect(c, ALIEN_DARK, (x - 12, top + 11, 3, 5))
        pygame.draw.rect(c, ALIEN_DARK, (x + 10, top + 11, 3, 5))
        pygame.draw.ellipse(c, ALIEN_DARK, (x - 10, top, 21, 20))
        pygame.draw.ellipse(c, ALIEN_BODY, (x - 9, top + 1, 19, 18))
        pygame.draw.ellipse(c, ALIEN_LIGHT, (x - 5, top + 11, 11, 6))
        pygame.draw.rect(c, BLACK, (x - 5, top + 5, 2, 3))
        pygame.draw.rect(c, BLACK, (x + 4, top + 5, 2, 3))
        c.set_at((x - 5, top + 5), WHITE)
        c.set_at((x + 4, top + 5), WHITE)
        pygame.draw.line(c, ALIEN_DARK, (x - 1, top + 9), (x + 1, top + 9))
        return (x, top - 6)

    def draw_ray(self, c, start, now):
        x0, y0 = start
        x1, y1 = CRYSTAL_X[self.aim], CRYSTAL_BASE_Y - 11
        n = max(abs(x1 - x0), abs(y1 - y0))
        firing = now - self.last_fire_ms < FIRE_PULSE_MS and self.trial > 0
        phase = (now // 70) % 5          # dashes flow toward the crystal
        for i in range(2, n + 1):
            px = round(x0 + (x1 - x0) * i / n)
            py = round(y0 + (y1 - y0) * i / n)
            if firing:
                c.set_at((px, py), RAY_BRIGHT)
                c.set_at((px, py + 1), RAY_COLOR)
            elif (i - phase) % 5 < 3:
                c.set_at((px, py), RAY_COLOR)

    def draw_crystal(self, c, side, now):
        cx, base = CRYSTAL_X[side], CRYSTAL_BASE_Y
        palette = CRYSTAL_NORMAL
        sparkle = False
        elapsed = now - self.flash_start_ms
        if side == self.flash_side and elapsed < FLASH_MS:
            if self.flash_kind == "success":
                palette, sparkle = CRYSTAL_BRIGHT, True
            elif (elapsed // 50) % 2 == 0:   # dim flicker
                palette = CRYSTAL_DIM

        top = (cx, base - CRYSTAL_H)
        bottom = (cx, base)
        left = (cx - CRYSTAL_HALF_W, base - 13)
        right = (cx + CRYSTAL_HALF_W, base - 13)
        pygame.draw.ellipse(c, SHADOW, (cx - 8, base - 2, 16, 4))
        pygame.draw.polygon(c, palette["light"], [top, bottom, left])
        pygame.draw.polygon(c, palette["dark"], [top, right, bottom])
        pygame.draw.polygon(c, palette["outline"], [top, right, bottom, left], 1)
        pygame.draw.line(c, palette["highlight"], (cx - 3, base - 13), (cx - 1, base - 17))

        if sparkle:
            cy = base - 11
            pygame.draw.circle(c, CRYSTAL_BRIGHT["dark"], (cx, cy), 14, 1)
            for sx, sy in ((cx - 10, cy - 9), (cx + 10, cy - 4), (cx - 9, cy + 6), (cx + 8, cy + 9)):
                for dx, dy in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
                    c.set_at((sx + dx, sy + dy), WHITE)

    def draw_meter(self, c):
        x, y, w, h = METER_RECT
        pygame.draw.rect(c, METER_FRAME, (x - 1, y - 1, w + 2, h + 2))
        pygame.draw.rect(c, METER_BG, (x, y, w, h))
        frac = min(1.0, self.shown_energy / METER_CAPACITY)
        fill_h = int(round(h * frac))
        if fill_h > 0:
            pygame.draw.rect(c, METER_FILL, (x, y + h - fill_h, w, fill_h))
            pygame.draw.line(c, METER_FILL_LIGHT, (x + 1, y + h - fill_h), (x + 1, y + h - 1))
            for yy in range(y + h - 4, y + h - fill_h, -4):
                pygame.draw.line(c, METER_SEG, (x, yy), (x + w - 1, yy))

    # ----- text & overlays -------------------------------------------------
    def font(self, size):
        if size not in self.fonts:
            self.fonts[size] = pygame.font.SysFont("couriernew,consolas,monospace", size, bold=True)
        return self.fonts[size]

    def text(self, msg, size, color, center):
        surf = self.font(size).render(msg, False, color)
        self.screen.blit(surf, surf.get_rect(center=(int(center[0]), int(center[1]))))

    def draw_panel(self, w, h):
        panel = pygame.Surface((w, h), pygame.SRCALPHA)
        panel.fill(PANEL_BG)
        rect = panel.get_rect(center=(SCREEN_W // 2, SCREEN_H // 2 - 60))
        self.screen.blit(panel, rect)
        pygame.draw.rect(self.screen, PANEL_BORDER, rect, 4)
        return rect

    def draw_start_screen(self, now):
        rect = self.draw_panel(600, 280)
        cx = rect.centerx
        self.text("ALIEN ENERGY FORAGER", 34, TITLE_COLOR, (cx, rect.top + 42))
        for i, line in enumerate(INSTRUCTIONS):
            self.text(line, 17, TEXT_COLOR, (cx, rect.top + 100 + i * 28))
        if (now // 500) % 2 == 0:
            self.text("Press SPACE to start", 22, RAY_COLOR, (cx, rect.bottom - 36))

    def draw_end_screen(self):
        rect = self.draw_panel(560, 220)
        cx = rect.centerx
        self.text("MISSION COMPLETE", 34, TITLE_COLOR, (cx, rect.top + 50))
        self.text(f"Total energy collected: {self.energy}", 22, TEXT_COLOR, (cx, rect.top + 115))
        self.text("Press ESC to exit", 18, RAY_COLOR, (cx, rect.bottom - 36))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Alien Energy Forager task")
    parser.add_argument("--out-dir", default=None,
                        help="folder for task_events.csv (default: a new folder in task/data/)")
    Game(parser.parse_args().out_dir).run()
