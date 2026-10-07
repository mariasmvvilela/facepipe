# Tasks

Recorders:
- `scripts\recording.py --task <script stem>`: player-paced tasks (space_shooter and future
  tasks like it)
- `scripts\recording_alien_forager.py`: Alien Energy Forager (automatic firing at a constant rate)

To be recordable, a task must:
- accept `--out-dir <folder>` and write `task_events.csv` there as events happen (standalone
  runs without it write to a new `task\data\<task>_YYYYMMDD_HHMMSS\` folder)
- use the event format below
- print `FACEPIPE_EVENT <name> <json>` lines on stdout (`emit()`). Required: `ready` when
  the start screen is up (recording starts) and `task_end` when the task is finished
  (otherwise the session is logged as aborted). Each logged event is also emitted (the
  recorder counts them for its progress line); `task_start` and `quit` are optional.

Run with the recorder (normal use): `python scripts\recording.py --task space_shooter --participant P01`
Run on its own (no recording): `python task\space_shooter.py`

### `task_events.csv`: one row per event
Columns: `timestamp, frame_idx, event, side`. The task writes `frame_idx` empty; the recorder
fills it with the nearest video frame after the session (standalone runs leave it empty).

| event | when | side |
|---|---|---|
| trial | a trial happens (shot / firing) | site of the trial |
| reward | that trial was rewarded (same timestamp) | site of the trial |
| fail | that trial was not rewarded (same timestamp) | site of the trial |
| switch | the participant leaves the current site | site they go to |
| system_flip | hidden: the active site flips (same timestamp as the trial that caused it) | newly active site |

## Space Shooter (current) — `space_shooter.py`

Two-option restless foraging task (hidden-state reversal) in Pygame. Copied from
`Documents/task_design/SpaceShooter.py`; the facepipe changes are the event log,
`emit()` calls and `--out-dir`. If the game is changed in `task_design`, copy it again
and re-add them.

SPACE = start / shoot, LEFT / RIGHT (hold) = move ship (2 s crossing), ESC = quit.
The game ends after 50 kills (`N_TRIALS`). `trial` = the player's shot; `switch` = the
ship starts moving off its position (if the player reverses mid-way and returns to the
same UFO, that is still logged as a switch).

## Alien Energy Forager (earlier) — `alien_energy_forager.py`

Earlier version of the task, used for session 20260928_161612 (folder
`session_20260928_161612_alien_energy_forager`, recorded before the event format: its
`task_events.csv` is the old one-row-per-trial log).

SPACE = start, LEFT / RIGHT = aim (instant), ESC = quit; the ray fires automatically
every second, 300 trials. `trial` = each automatic firing; `switch` = the player re-aims.
