# Tasks

## Space Shooter (current) — `space_shooter.py`

Two-option restless foraging task (hidden-state reversal) in Pygame. Copied from
`Documents/task_design/SpaceShooter.py`; the only change is `emit()` calls that print
`FACEPIPE_EVENT <name> <json>` lines on stdout for `scripts/record_space_shooter.py`.
If the game is changed in `task_design`, copy it again and re-add the `emit()` calls.

Run with the recorder (normal use): `python scripts\record_space_shooter.py`
Run on its own (no recording): `python task\space_shooter.py`

Output: one row per shot appended to `task\data\space_shooter_data.csv`
(the recorder copies this run's rows into the session folder as `task_events.csv`).

### Controls
SPACE = start / shoot, LEFT / RIGHT (hold) = move ship (2 s crossing), ESC = quit.
The game ends after 50 kills (`N_TRIALS`).

### Output CSV columns
| column | meaning |
|---|---|
| trial | shot number, from 1 |
| choice | side shot at (left / right) |
| active_ufo | hidden active side at the time of the shot |
| outcome | reward / failure |
| switch_occurred | whether the active side flipped after this shot |
| rt_ms | ms since the previous shot (task start for trial 1) |
| timestamp | wall-clock time of the shot (ms precision) |
| time_on_choice_ms | ms since the ship arrived at its current position |
| session_id | task run ID (game launch time), distinguishes appended runs |

### Recorder event markers (`task_markers.csv`)
`task_start`, `depart` (ship leaves a position), `arrive`, `shot`, `task_end`, `quit`,
each with wall-clock timestamp and nearest video frame.

## Alien Energy Forager (earlier) — `alien_energy_forager.py`
Earlier version of the task, used for session 20260928_161612. Output:
`task\data\foraging_data.csv`; record it manually with `scripts\record_session.py`.
