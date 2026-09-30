"""Record a facepipe session: run the Space Shooter task and film the face while it runs.

1. Opens the webcam and shows a preview to check framing (SPACE = launch the game,
   ESC = cancel).
2. Launches task/space_shooter.py. The game prints a FACEPIPE_EVENT line on stdout
   for each task event; this script listens to them.
3. Recording starts when SPACE is pressed on the game's start screen. The last
   PRE_ROLL_S of camera frames before that are kept, so the video begins slightly
   before the task (baseline face, and no risk of missing the start).
4. Recording stops POST_ROLL_S after the game-over screen appears (so the last
   outcome's reaction is filmed), or straight away if the game is closed early.

Writes a session folder ready for the pipeline, raw_video/session_YYYYMMDD_HHMMSS/
(named after the first recorded frame, same format as the task's session_id):

    recording_YYYY-MM-DD_HH-MM-SS.avi             the video (start time in the name, st4 reads it)
    recording_YYYY-MM-DD_HH-MM-SS_frametimes.csv  frame_idx, wall-clock time of each frame
                                                  (taken right after cap.read())
    task_events.csv                               this run's rows of the game's trial CSV
                                                  (one row per shot)
    task_markers.csv                              every game event (task_start, depart, arrive,
                                                  shot, task_end, ...) with its wall-clock time
                                                  and nearest video frame
    session_info.txt                              timing, camera and task details

All times are the same PC wall clock (datetime.now() in both processes), so trial
timestamps, markers and frame times line up directly.

The preview is closed while the game runs so the participant only sees the game;
progress is printed in this console. Ctrl+C here stops and saves the recording and
closes the game.

Run inside the facepipe conda environment (it has pygame):
    python scripts\\record_space_shooter.py
    python scripts\\record_space_shooter.py --participant P01 --notes "glasses off"
"""
import argparse
import bisect
import csv
import hashlib
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime

# Also does the facepipe-env check (cv2 + mediapipe) and exits with instructions if it fails.
from record_session import (CAMERA_INDICES, GAP_FACTOR, KEY_ESC, MAX_CONSECUTIVE_FAILS,
                            REQUEST_FPS, REQUEST_HEIGHT, REQUEST_WIDTH, open_camera)
import cv2

# --- Settings -----------------------------------------------------------------
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_VIDEO_DIR = os.path.join(PROJECT_DIR, "raw_video")
GAME_SCRIPT = os.path.join(PROJECT_DIR, "task", "space_shooter.py")

PRE_ROLL_S = 1.0             # camera frames kept from before the task starts
POST_ROLL_S = 2.0            # keep recording after the game-over screen appears
MAX_DURATION_S = 20 * 60     # safety cap
PROGRESS_EVERY_S = 10
EVENT_PREFIX = "FACEPIPE_EVENT"

WINDOW = "facepipe framing check (SPACE = launch game, ESC = cancel)"
MARKER_COLUMNS = ["event", "timestamp", "frame_idx", "trial", "side", "choice", "outcome", "kills"]


def read_game_output(stream, events):
    """Thread: parse the game's stdout; FACEPIPE_EVENT lines go to the queue, the rest is echoed."""
    for line in stream:
        line = line.rstrip()
        if line.startswith(EVENT_PREFIX + " "):
            try:
                _, name, payload = line.split(" ", 2)
                events.put((name, json.loads(payload)))
                continue
            except ValueError:
                pass
        if line:
            print("  [game] " + line)
    events.put(("_stdout_closed", {}))


def write_frames(frames_q, writer, frametimes_file):
    """Thread: encode frames off the capture loop so frame timestamps stay accurate."""
    idx = 0
    while True:
        item = frames_q.get()
        if item is None:
            return
        frame, stamp = item
        writer.write(frame)
        frametimes_file.write("{},{}\n".format(idx, stamp.isoformat(timespec="microseconds")))
        idx += 1


def framing_check(cap):
    """Live preview until SPACE/Enter (True) or ESC / window closed (False)."""
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    fails = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            fails += 1
            if fails >= MAX_CONSECUTIVE_FAILS:
                print("ERROR: camera stopped delivering frames.")
                return False
            continue
        fails = 0
        preview = frame.copy()
        cv2.putText(preview, "Centre the face, then press SPACE to launch the game",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow(WINDOW, preview)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord(" "), 13):
            return True
        try:
            closed = cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1
        except cv2.error:
            closed = True
        if key == KEY_ESC or closed:
            return False


def file_sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--participant", default="", help="participant ID, saved in session_info.txt")
    parser.add_argument("--notes", default="", help="free-text notes, saved in session_info.txt")
    args = parser.parse_args()

    if not os.path.isfile(GAME_SCRIPT):
        sys.exit("ERROR: game not found: {}".format(GAME_SCRIPT))

    # --- Step 1: camera + framing check ---------------------------------------------
    cap, cam_index = open_camera()
    if cap is None:
        print("\nERROR: could not open a camera (tried indices {}).".format(CAMERA_INDICES))
        print("Check the USB webcam is plugged in and not in use by another app "
              "(Teams, Zoom, Camera app), then try again.\n")
        sys.exit(1)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    reported_fps = cap.get(cv2.CAP_PROP_FPS)
    writer_fps = reported_fps if 1 <= reported_fps <= 120 else REQUEST_FPS
    print("\nCamera {} opened: {}x{} @ {:.2f} fps (requested {}x{} @ {})".format(
        cam_index, width, height, reported_fps, REQUEST_WIDTH, REQUEST_HEIGHT, REQUEST_FPS))
    print("Check framing in the preview window, then press SPACE to launch the game (ESC = cancel).")

    if not framing_check(cap):
        cap.release()
        cv2.destroyAllWindows()
        print("\nCancelled. Nothing recorded.")
        return
    cv2.destroyAllWindows()
    cv2.waitKey(1)

    # --- Step 2: launch the game --------------------------------------------------
    env = dict(os.environ, PYGAME_HIDE_SUPPORT_PROMPT="1")
    game = subprocess.Popen([sys.executable, "-u", GAME_SCRIPT], cwd=os.path.dirname(GAME_SCRIPT),
                            stdout=subprocess.PIPE, stderr=None, text=True, bufsize=1, env=env)
    events = queue.Queue()
    threading.Thread(target=read_game_output, args=(game.stdout, events), daemon=True).start()
    print("\nGame launched. Waiting for SPACE on the game's start screen "
          "(click the game window if it doesn't have focus).")

    # --- Step 3: capture loop -----------------------------------------------------------
    pre_roll = deque(maxlen=max(1, round(PRE_ROLL_S * writer_fps)))
    all_events = []            # (name, fields) in arrival order
    game_info = {}
    recording = False
    status = None              # completed / aborted / ...
    stop_at = None
    frames_q = writer = frametimes_file = writer_thread = None
    session_dir = video_stem = None
    frame_times = []           # datetime of every written frame
    t_mono = []                # time.time() of every written frame, for fps / gaps
    read_fails = consecutive_fails = 0
    next_progress = PROGRESS_EVERY_S
    last_shot = {}

    try:
        while True:
            ok, frame = cap.read()
            now = time.time()
            stamp = datetime.now()
            if not ok:
                consecutive_fails += 1
                read_fails += recording
                if consecutive_fails >= MAX_CONSECUTIVE_FAILS:
                    print("\nERROR: camera stopped delivering frames.")
                    status = "camera stopped delivering frames"
                    break
            else:
                consecutive_fails = 0
                if recording:
                    frames_q.put((frame, stamp))
                    frame_times.append(stamp)
                    t_mono.append(now)
                else:
                    pre_roll.append((frame, stamp, now))

            # Game events
            while not events.empty():
                name, fields = events.get()
                if name == "_stdout_closed":
                    continue
                all_events.append((name, fields))
                if name == "ready":
                    game_info = fields
                    print("Task session {} ready ({} kills to finish).".format(
                        fields.get("session_id"), fields.get("n_trials")))
                elif name == "task_start" and not recording:
                    # Session folder and file names come from the first (pre-roll) frame.
                    first_stamp = pre_roll[0][1] if pre_roll else stamp
                    session_dir = os.path.join(RAW_VIDEO_DIR, "session_" + first_stamp.strftime("%Y%m%d_%H%M%S"))
                    os.makedirs(session_dir, exist_ok=False)
                    video_stem = "recording_" + first_stamp.strftime("%Y-%m-%d_%H-%M-%S")
                    video_path = os.path.join(session_dir, video_stem + ".avi")
                    writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"XVID"),
                                             writer_fps, (width, height))
                    if not writer.isOpened():
                        shutil.rmtree(session_dir, ignore_errors=True)
                        session_dir = None
                        print("ERROR: could not create video file: {}".format(video_path))
                        status = "could not create video file"
                        raise SystemExit
                    frametimes_file = open(os.path.join(session_dir, video_stem + "_frametimes.csv"), "w")
                    frametimes_file.write("frame_idx,wall_clock_timestamp\n")
                    frames_q = queue.Queue()
                    writer_thread = threading.Thread(target=write_frames,
                                                     args=(frames_q, writer, frametimes_file))
                    writer_thread.start()
                    for f, s, m in pre_roll:
                        frames_q.put((f, s))
                        frame_times.append(s)
                        t_mono.append(m)
                    pre_roll.clear()
                    recording = True
                    print("Task started -> recording to raw_video\\{}\\".format(os.path.basename(session_dir)))
                elif name == "shot":
                    last_shot = fields
                elif name == "task_end" and recording and stop_at is None:
                    stop_at = now + POST_ROLL_S
                    print("Game over ({} shots, {} kills). Stopping in {:.0f} s ...".format(
                        fields.get("trials"), fields.get("kills"), POST_ROLL_S))
                elif name == "quit" and recording and stop_at is None:
                    status = "aborted: game closed before the end"

            if stop_at is not None and now >= stop_at:
                status = "completed"
                break
            if status is not None:
                break
            if game.poll() is not None and events.empty():
                if stop_at is not None:      # closed during the post-roll: the task was finished
                    status = "completed"
                elif recording:
                    status = "aborted: game exited (code {}) before the end".format(game.returncode)
                break

            if recording:
                elapsed = now - t_mono[0]
                if elapsed >= next_progress:
                    print("  recording {:4d} s | {} frames | shot {} | kills {}/{}".format(
                        int(elapsed), len(frame_times), last_shot.get("trial", 0),
                        last_shot.get("kills", 0), game_info.get("n_trials", "?")))
                    next_progress += PROGRESS_EVERY_S
                if elapsed >= MAX_DURATION_S:
                    print("\nReached the {}-minute safety cap, stopping.".format(MAX_DURATION_S // 60))
                    status = "max duration reached"
                    break
    except KeyboardInterrupt:
        print("\nCtrl+C received, stopping and saving.")
        status = "Ctrl+C"
    except SystemExit:
        pass
    finally:
        if writer_thread is not None:
            frames_q.put(None)
            writer_thread.join()
        if writer is not None:
            writer.release()
        if frametimes_file is not None:
            frametimes_file.close()
        cap.release()

    if not recording:
        if game.poll() is None:
            game.terminate()
        print("\nThe task was never started. Nothing recorded.")
        return
    if status != "completed" and game.poll() is None:
        game.terminate()   # Ctrl+C / camera failure: don't leave the game running
    if not frame_times:
        shutil.rmtree(session_dir, ignore_errors=True)
        print("\nNo frames were captured. Session folder deleted.")
        return

    # --- Step 4: task CSV (this run only) ---------------------------------------------------
    task_session = game_info.get("session_id")
    data_path = game_info.get("data_path")
    n_rows = 0
    if data_path and os.path.isfile(data_path):
        with open(data_path, newline="") as f:
            reader = csv.DictReader(f)
            rows = [r for r in reader if r["session_id"] == task_session]
            columns = reader.fieldnames
        with open(os.path.join(session_dir, "task_events.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=columns)
            w.writeheader()
            w.writerows(rows)
        n_rows = len(rows)
        csv_status = "{} trials copied from {}".format(n_rows, os.path.relpath(data_path, PROJECT_DIR))
    else:
        csv_status = "NOT FOUND ({})".format(data_path)
        print("WARNING: task CSV not found: {}".format(data_path))

    # --- Step 5: event markers with their video frame --------------------------------------------
    def nearest_frame(t):
        """Index of the frame closest in time to t, or "" if t is outside the video."""
        if t < frame_times[0] or t > frame_times[-1]:
            return ""
        i = bisect.bisect_left(frame_times, t)
        if i == 0:
            return 0
        return i if frame_times[i] - t < t - frame_times[i - 1] else i - 1

    with open(os.path.join(session_dir, "task_markers.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MARKER_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for name, fields in all_events:
            if name == "ready":
                continue
            t = datetime.fromisoformat(fields["t"])
            w.writerow({"event": name, "timestamp": fields["t"], "frame_idx": nearest_frame(t), **fields})

    # --- Step 6: session info --------------------------------------------------------------
    n_frames = len(frame_times)
    duration = t_mono[-1] - t_mono[0]
    actual_fps = (n_frames - 1) / duration if duration > 0 else 0.0
    intervals = [b - a for a, b in zip(t_mono, t_mono[1:])]
    timing_gaps = sum(dt > GAP_FACTOR / writer_fps for dt in intervals)
    first = {name: fields["t"] for name, fields in reversed(all_events)}
    start_dt = frame_times[0]
    info_lines = [
        "date: " + start_dt.strftime("%Y-%m-%d"),
        "time: " + start_dt.strftime("%H-%M-%S"),
        "duration_seconds: {:.2f}".format(duration),
        "total_frames: {}".format(n_frames),
        "actual_fps: {:.3f}".format(actual_fps),
        "actual_resolution: {}x{}".format(width, height),
        "camera_index: {}".format(cam_index),
        "participant: " + args.participant,
        "notes: " + args.notes,
        "",
        "# task",
        "task: space_shooter ({})".format(os.path.relpath(GAME_SCRIPT, PROJECT_DIR)),
        "task_script_sha256: " + file_sha256(GAME_SCRIPT),
        "task_session_id: {}".format(task_session),
        "task_status: " + status,
        "task_trials_logged: {}".format(n_rows),
        "task_kills: {} / {}".format(last_shot.get("kills", 0), game_info.get("n_trials")),
        "task_start: {}".format(first.get("task_start")),
        "task_end: {}".format(first.get("task_end", "not reached")),
        "task_parameters: n_trials={} p_reward={} p_switch={} travel_time_ms={} random_seed={}".format(
            *(game_info.get(k) for k in ("n_trials", "p_reward", "p_switch", "travel_time_ms", "random_seed"))),
        "",
        "# technical",
        "first_frame: " + frame_times[0].isoformat(timespec="microseconds"),
        "last_frame: " + frame_times[-1].isoformat(timespec="microseconds"),
        "pre_roll_seconds: {}".format(PRE_ROLL_S),
        "post_roll_seconds: {}".format(POST_ROLL_S),
        "video_container_fps: {:.3f}".format(writer_fps),
        "camera_reported_fps: {:.3f}".format(reported_fps),
        "failed_frame_reads: {}".format(read_fails),
        "frame_timing_gaps: {}".format(timing_gaps),
        "stop_reason: " + status,
    ]
    with open(os.path.join(session_dir, "session_info.txt"), "w") as f:
        f.write("\n".join(info_lines) + "\n")

    # --- Step 7: summary --------------------------------------------------------------------------
    rel_session = os.path.relpath(session_dir, PROJECT_DIR)
    print("\nSession saved to: {}\\  ({})".format(rel_session, status))
    print("  {}.avi - {} frames, {:.1f} s, {:.2f} fps".format(video_stem, n_frames, duration, actual_fps))
    print("  {}_frametimes.csv - {} frame timestamps".format(video_stem, n_frames))
    print("  task_events.csv  - {}".format(csv_status))
    print("  task_markers.csv - {} events".format(sum(1 for n, _ in all_events if n != "ready")))
    print("  session_info.txt - written")
    if read_fails or timing_gaps:
        print("  note: {} failed reads, {} timing gaps (logged in session_info.txt)".format(
            read_fails, timing_gaps))
    if abs(actual_fps - writer_fps) > 1:
        print("  note: measured fps ({:.2f}) differs from the video file's fps ({:.2f}); "
              "use actual_fps / the frametimes file for timing.".format(actual_fps, writer_fps))
    print("\nNext step:")
    print("  python scripts\\st1_mediapipe_landmarks.py --session {}".format(rel_session))

    if game.poll() is None:
        print("\nWaiting for the game window to close (press ESC in the game) ...")
        try:
            game.wait()
        except KeyboardInterrupt:
            game.terminate()


if __name__ == "__main__":
    main()
