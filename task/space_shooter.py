"""
Space Shooter
=============
A two-option restless foraging task (hidden-state reversal task) in Pygame.

A spaceship hovers above a moon and defends it from two identical UFOs, one
top-left and one top-right. At any moment exactly one of them is secretly
"active". When the player shoots (SPACE) at the UFO directly above the ship:
    - active UFO   -> kill with probability P_REWARD
    - inactive UFO -> never a kill
Each shot at the active UFO flips it with probability P_SWITCH (independently
of the kill), and the other UFO silently becomes active. Shooting the inactive
UFO changes nothing, so once the active UFO flips it stays that way until the
player goes over and shoots the other one.

The ship moves between the two positions while LEFT / RIGHT is held; a full
crossing takes TRAVEL_TIME_MS. Shooting is only possible once the ship has fully
arrived at a position.

Events (trial, reward, fail, switch, system_flip) are written as they happen to
task_events.csv in the output folder: the folder given by --out-dir (the recorder
passes the session folder), or, when run standalone, a new
task/data/space_shooter_YYYYMMDD_HHMMSS/ folder.

Controls: SPACE = start / shoot, LEFT/RIGHT (hold) = move, ESC = quit
(data is saved after every event).
Requires: pygame. numpy is used for sound generation if installed; otherwise a
standard-library fallback is used.

facepipe: copied from Documents/task_design/SpaceShooter.py. Additions: the event
log above, --out-dir, and emit(): one "FACEPIPE_EVENT <name> <json>" line on stdout
per event (ready, task_start, every logged event, task_end, quit), which
scripts/recording.py reads to time the webcam recording. The game still runs on its own.
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
import numpy as np  # optional, used for sound generation if installed
# ---------------------------------------------------------------------------
# Task parameters
# ---------------------------------------------------------------------------
N_TRIALS = 50             # game ends after this many kills
P_REWARD = 0.7            # kill probability when shooting the active UFO
P_SWITCH = 0.3            # probability a shot at the active UFO flips which is active
TRAVEL_TIME_MS = 2000     # time to move between positions
RANDOM_SEED = None        # set to an int for a reproducible reward/switch sequence

# ---------------------------------------------------------------------------
# Display / timing
# ---------------------------------------------------------------------------
SCREEN_W, SCREEN_H = 800, 600
PIXEL = 4                              # size of one "art pixel" on screen
CANVAS_W, CANVAS_H = SCREEN_W // PIXEL, SCREEN_H // PIXEL
FPS = 60
N_STARS = 90
FLASH_MS = 200                         # UFO feedback flash duration
FIRE_PULSE_MS = 150                    # beam visible after a shot
END_DELAY_MS = 800                     # pause after last shot before end screen
BOB_PERIOD_S = 1.6                     # UFO hover bob cycle
BOB_AMPLITUDE = 1.5                    # in art pixels
LIGHT_BLINK_MS = 300                   # UFO rim-light blink period

# The meter shows aliens (shots) remaining, full at the start
METER_CAPACITY = N_TRIALS

# ---------------------------------------------------------------------------
# Data logging
# ---------------------------------------------------------------------------
TASK_NAME = "space_shooter"
DATA_DIR = "data"                     # standalone runs: data/space_shooter_YYYYMMDD_HHMMSS/
DATA_FILE = "task_events.csv"
# One row per event. frame_idx is left empty here; the recorder fills it in after
# the session (nearest video frame). Events:
#   trial        a trial happens at `side` (the player shoots)
#   reward/fail  that trial's outcome (same timestamp as the trial)
#   switch       the participant leaves the current site; side = the site they go to
#                (the ship starts moving off its position)
#   system_flip  hidden: the active site flips; side = the newly active site
CSV_COLUMNS = ["timestamp", "frame_idx", "event", "side"]

INSTRUCTIONS = [
    "Defend your planet from alien invaders!",
    "Press LEFT / RIGHT to move your ship.",
    "Press SPACE to fire when you arrive.",
    "One UFO is active — figure out which one.",
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

SHIP_BODY = (220, 225, 240)
SHIP_DARK = (70, 80, 120)
SHIP_WING = (232, 92, 124)
COCKPIT = (90, 220, 240)
FLAME = (255, 170, 60)
FLAME_CORE = (255, 240, 120)
BLACK = (10, 10, 16)
WHITE = (255, 255, 255)

UFO_NORMAL = {"body": (150, 160, 185), "under": (90, 98, 125), "outline": (35, 38, 60),
              "dome": (120, 230, 170), "dome_light": (210, 255, 230)}
UFO_BRIGHT = {"body": (255, 255, 255), "under": (210, 220, 240), "outline": (150, 170, 220),
              "dome": (230, 255, 240), "dome_light": (255, 255, 255)}
UFO_DIM = {"body": (70, 75, 95), "under": (45, 48, 65), "outline": (20, 22, 35),
           "dome": (50, 95, 75), "dome_light": (80, 120, 100)}
UFO_LIGHT_ON = (255, 240, 120)
UFO_LIGHT_OFF = (150, 90, 60)

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
GROUND_RECT = (-50, 120, 300, 150)     # big ellipse -> curved moon horizon
CRATERS = [(20, 132, 18, 5), (150, 138, 20, 6), (90, 142, 14, 4),
           (166, 134, 10, 3), (6, 140, 10, 3), (60, 128, 8, 2)]
UFO_X = {"left": 56, "right": 144}
UFO_Y = 40                             # centre of the saucer body
SHIP_BASE_Y = 116                      # bottom of the ship hull
METER_RECT = (184, 34, 9, 80)          # x, y, w, h


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
        pygame.display.set_caption("Space Shooter")
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
            y = star_rng.randrange(0, 125)
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
        self.aim = self.rng.choice(["left", "right"])   # None while between positions
        self.active = self.rng.choice(["left", "right"])
        self.ship_pos = 0.0 if self.aim == "left" else 1.0   # 0 = left, 1 = right
        self.moving = False
        self.trial = 0
        self.kills = 0
        self.shown_remaining = float(N_TRIALS)
        self.last_fire_ms = 0
        self.aim_since_ms = 0
        self.flash_side = None
        self.flash_kind = None
        self.flash_start_ms = -10_000
        emit("ready", session_id=self.session_id, data_path=self.data_path,
             n_trials=N_TRIALS, p_reward=P_REWARD, p_switch=P_SWITCH,
             travel_time_ms=TRAVEL_TIME_MS, random_seed=RANDOM_SEED, fps=FPS,
             start_side=self.aim, start_active=self.active)

    # ----- task logic ------------------------------------------------------
    def start_task(self, now):
        self.state = "task"
        self.last_fire_ms = now
        self.aim_since_ms = now
        emit("task_start", session_id=self.session_id, side=self.aim)

    def update_travel(self, dt_ms, now):
        """Slide the ship while LEFT/RIGHT is held; a full crossing takes TRAVEL_TIME_MS."""
        keys = pygame.key.get_pressed()
        direction = keys[pygame.K_RIGHT] - keys[pygame.K_LEFT]
        if direction == 0 or self.kills >= N_TRIALS:
            self.moving = False
            return
        old = self.ship_pos
        self.ship_pos = max(0.0, min(1.0, old + direction * dt_ms / TRAVEL_TIME_MS))
        self.moving = self.ship_pos != old
        if self.moving:
            if self.aim is not None:
                # Leaving a position is the switch decision; the ship can only head
                # to the other position (reversing mid-way returns to the same one).
                self.log_event("switch", "right" if direction > 0 else "left", datetime.now())
            self.aim = None
            if self.ship_pos in (0.0, 1.0):
                self.aim = "left" if self.ship_pos == 0.0 else "right"
                self.aim_since_ms = now

    def fire(self, now):
        self.trial += 1
        choice = self.aim
        active = self.active
        success = choice == active and self.rng.random() < P_REWARD
        # Only shooting the active UFO can flip it; the inactive one never
        # becomes active on its own, so the player must go to the other UFO.
        switched = choice == active and self.rng.random() < P_SWITCH

        t = datetime.now()
        self.log_event("trial", choice, t)
        self.log_event("reward" if success else "fail", choice, t)
        if switched:
            self.log_event("system_flip", other(active), t)

        if switched:
            self.active = other(active)
        if success:
            self.kills += 1
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
            dt_ms = self.clock.tick(FPS)
            dt = dt_ms / 1000.0
            now = pygame.time.get_ticks()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.quit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.quit()
                    elif self.state == "start" and event.key == pygame.K_SPACE:
                        self.start_task(now)
                    elif (self.state == "task" and event.key == pygame.K_SPACE
                          and self.aim is not None and self.kills < N_TRIALS):
                        self.fire(now)

            if self.state == "task":
                self.update_travel(dt_ms, now)
                if self.kills >= N_TRIALS and now - self.last_fire_ms >= END_DELAY_MS:
                    self.state = "end"
                    emit("task_end", trials=self.trial, kills=self.kills, n_rewards=self.kills)

            remaining = N_TRIALS - self.kills
            self.shown_remaining += (remaining - self.shown_remaining) * min(1.0, dt * 8)
            self.draw(now)

    def quit(self):
        emit("quit", state=self.state, trials=self.trial, kills=self.kills, n_rewards=self.kills)
        pygame.quit()
        sys.exit()

    # ----- drawing ---------------------------------------------------------
    def draw(self, now):
        c = self.canvas
        c.fill(SKY)
        self.draw_stars(c)
        self.draw_ground(c)
        ship_x = round(UFO_X["left"] + (UFO_X["right"] - UFO_X["left"]) * self.ship_pos)
        if self.state == "task":
            self.draw_ray(c, (ship_x, SHIP_BASE_Y - 21), now)
        for side in ("left", "right"):
            self.draw_ufo(c, side, now)
        self.draw_ship(c, ship_x, now)
        self.draw_meter(c)

        pygame.transform.scale(c, (SCREEN_W, SCREEN_H), self.screen)
        mx, my, mw, _ = METER_RECT
        self.text("ALIENS", 18, TEXT_COLOR, ((mx + mw / 2) * PIXEL, my * PIXEL - 34))
        self.text("LEFT", 18, TEXT_COLOR, ((mx + mw / 2) * PIXEL, my * PIXEL - 16))

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

    def draw_ship(self, c, x, now):
        """Draw the player's spaceship with its hull bottom at SHIP_BASE_Y."""
        base = SHIP_BASE_Y
        # Engine flame: flickers, longer while travelling
        flame_len = (4 if self.moving else 2) + (now // 60) % 2
        pygame.draw.rect(c, FLAME, (x - 2, base, 5, flame_len))
        pygame.draw.rect(c, FLAME_CORE, (x - 1, base, 3, flame_len - 1))

        for s in (-1, 1):   # wings
            wing = [(x + 4 * s, base - 10), (x + 10 * s, base - 3),
                    (x + 10 * s, base), (x + 4 * s, base - 2)]
            pygame.draw.polygon(c, SHIP_WING, wing)
            pygame.draw.polygon(c, SHIP_DARK, wing, 1)

        hull = [(x, base - 20), (x + 4, base - 13), (x + 4, base - 2),
                (x - 4, base - 2), (x - 4, base - 13)]
        pygame.draw.polygon(c, SHIP_BODY, hull)
        pygame.draw.polygon(c, SHIP_DARK, hull, 1)
        pygame.draw.rect(c, SHIP_DARK, (x - 3, base - 2, 7, 2))      # nozzle
        pygame.draw.ellipse(c, SHIP_DARK, (x - 3, base - 15, 7, 7))
        pygame.draw.ellipse(c, COCKPIT, (x - 2, base - 14, 5, 5))
        c.set_at((x - 1, base - 13), WHITE)

    def draw_ray(self, c, start, now):
        """Brief vertical beam from the ship's nose to the UFO above it."""
        if not (now - self.last_fire_ms < FIRE_PULSE_MS and self.trial > 0):
            return
        x, y0 = start
        y1 = UFO_Y + 5 + self.ufo_bob(now)
        for y in range(y1, y0 + 1):
            c.set_at((x, y), RAY_BRIGHT)
            c.set_at((x - 1, y), RAY_COLOR)
            c.set_at((x + 1, y), RAY_COLOR)

    def ufo_bob(self, now):
        # Same phase for both UFOs so they always look identical
        return round(BOB_AMPLITUDE * math.sin(2 * math.pi * (now / 1000.0) / BOB_PERIOD_S))

    def draw_ufo(self, c, side, now):
        cx, cy = UFO_X[side], UFO_Y + self.ufo_bob(now)
        palette = UFO_NORMAL
        sparkle = False
        elapsed = now - self.flash_start_ms
        if side == self.flash_side and elapsed < FLASH_MS:
            if self.flash_kind == "success":
                palette, sparkle = UFO_BRIGHT, True
            elif (elapsed // 50) % 2 == 0:   # dim flicker
                palette = UFO_DIM

        # Dome (lower half is covered by the saucer body)
        pygame.draw.ellipse(c, palette["outline"], (cx - 8, cy - 11, 17, 14))
        pygame.draw.ellipse(c, palette["dome"], (cx - 7, cy - 10, 15, 12))
        pygame.draw.line(c, palette["dome_light"], (cx - 4, cy - 6), (cx - 2, cy - 8))
        # Saucer body
        pygame.draw.ellipse(c, palette["outline"], (cx - 10, cy + 1, 21, 7))
        pygame.draw.ellipse(c, palette["under"], (cx - 9, cy + 2, 19, 5))
        pygame.draw.ellipse(c, palette["outline"], (cx - 17, cy - 5, 35, 11))
        pygame.draw.ellipse(c, palette["body"], (cx - 16, cy - 4, 33, 9))
        pygame.draw.line(c, palette["under"], (cx - 13, cy + 2), (cx + 13, cy + 2))
        # Rim lights, blinking in the same pattern on both UFOs
        blink = now // LIGHT_BLINK_MS
        for i, dx in enumerate((-12, -6, 0, 6, 12)):
            color = UFO_LIGHT_ON if (i + blink) % 2 == 0 else UFO_LIGHT_OFF
            pygame.draw.rect(c, color, (cx + dx, cy - 1, 2, 2))

        if sparkle:
            pygame.draw.circle(c, UFO_BRIGHT["under"], (cx, cy), 21, 1)
            for sx, sy in ((cx - 16, cy - 12), (cx + 17, cy - 7), (cx - 14, cy + 10), (cx + 12, cy + 12)):
                for dx, dy in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
                    c.set_at((sx + dx, sy + dy), WHITE)

    def draw_meter(self, c):
        x, y, w, h = METER_RECT
        pygame.draw.rect(c, METER_FRAME, (x - 1, y - 1, w + 2, h + 2))
        pygame.draw.rect(c, METER_BG, (x, y, w, h))
        frac = max(0.0, min(1.0, self.shown_remaining / METER_CAPACITY))
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
        self.text("SPACE SHOOTER", 34, TITLE_COLOR, (cx, rect.top + 42))
        for i, line in enumerate(INSTRUCTIONS):
            self.text(line, 17, TEXT_COLOR, (cx, rect.top + 100 + i * 28))
        if (now // 500) % 2 == 0:
            self.text("Press SPACE to start", 22, RAY_COLOR, (cx, rect.bottom - 36))

    def draw_end_screen(self):
        rect = self.draw_panel(560, 220)
        cx = rect.centerx
        self.text("PLANET DEFENDED", 34, TITLE_COLOR, (cx, rect.top + 50))
        self.text(f"Aliens destroyed: {self.kills} / {N_TRIALS}", 22, TEXT_COLOR, (cx, rect.top + 115))
        self.text("Press ESC to exit", 18, RAY_COLOR, (cx, rect.bottom - 36))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Space Shooter task")
    parser.add_argument("--out-dir", default=None,
                        help="folder for task_events.csv (default: a new folder in task/data/)")
    Game(parser.parse_args().out_dir).run()
