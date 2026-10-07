"""Record a facepipe session for the Alien Energy Forager task.

Same recording as scripts/recording.py (framing check, recording from the task's
"ready" until the task window closes, raw_data/session_YYYYMMDD_HHMMSS_alien_energy_forager/),
but specific to this task's design: the ray fires automatically every
TRIAL_INTERVAL_MS for a fixed N_TRIALS, so the session length is known in advance
and the participant's only choice is where to aim.

    task_events.csv   timestamp, frame_idx, event, side
                      trial        automatic firing at the crystal being aimed at
                      reward/fail  that firing's outcome
                      switch       the participant re-aims (side = new crystal)
                      system_flip  hidden: the active crystal flips (side = new active one)

Forager-specific extras:
  * progress shows trial X / N_TRIALS and the time left at the fixed firing rate
  * session_info.txt gets the expected duration, the measured interval between
    firings (to check the constant rate held; dropped frames in the game show up as
    late firings), and the number of trials per visit to a crystal

    conda activate facepipe
    python scripts\\recording_alien_forager.py --participant P01
    python scripts\\recording_alien_forager.py --participant P01 --notes "glasses off"
"""
import argparse
import statistics
from datetime import datetime

from recording import run_session   # also does the facepipe-env check

TASK_NAME = "alien_energy_forager"
LATE_FIRING_FACTOR = 1.5   # a firing interval > 1.5x TRIAL_INTERVAL_MS counts as late


def progress(counts, task_info, elapsed):
    n_trials = task_info.get("n_trials")
    interval_s = task_info.get("trial_interval_ms", 0) / 1000
    line = "trial {} / {} | rewards {} | switches {}".format(
        counts["trial"], n_trials, counts["reward"], counts["switch"])
    if n_trials and interval_s:
        line += " | ~{:.0f} s left".format(max(0, n_trials - counts["trial"]) * interval_s)
    return line


def extra_info(rows, task_info):
    interval_ms = task_info.get("trial_interval_ms")
    n_trials = task_info.get("n_trials")
    lines = ["", "# alien_energy_forager"]
    if interval_ms and n_trials:
        lines.append("expected_task_duration_seconds: {:.0f}".format(n_trials * interval_ms / 1000))

    trial_times = [datetime.fromisoformat(r["timestamp"]) for r in rows if r["event"] == "trial"]
    gaps_ms = [(b - a).total_seconds() * 1000 for a, b in zip(trial_times, trial_times[1:])]
    if gaps_ms:
        late = sum(g > LATE_FIRING_FACTOR * interval_ms for g in gaps_ms) if interval_ms else "?"
        lines += [
            "firing_interval_ms: mean {:.1f} | min {:.1f} | max {:.1f} (target {})".format(
                statistics.mean(gaps_ms), min(gaps_ms), max(gaps_ms), interval_ms),
            "late_firings: {}".format(late),
        ]

    # Trials per visit: consecutive trials at the same crystal
    sides = [r["side"] for r in rows if r["event"] == "trial"]
    visits = []
    for side in sides:
        if visits and visits[-1][0] == side:
            visits[-1][1] += 1
        else:
            visits.append([side, 1])
    if visits:
        lengths = [n for _, n in visits]
        lines.append("visits: {} | trials per visit: mean {:.1f} | max {}".format(
            len(visits), statistics.mean(lengths), max(lengths)))
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--participant", default="", help="participant ID (e.g. P01), saved in session_info.txt")
    parser.add_argument("--notes", default="", help="free-text notes, saved in session_info.txt")
    args = parser.parse_args()
    run_session(TASK_NAME, args.participant, args.notes, progress=progress, extra_info=extra_info)


if __name__ == "__main__":
    main()
