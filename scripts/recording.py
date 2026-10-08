"""Record a facepipe session for a player-paced task (space_shooter and similar).

For alien_energy_forager (automatic firing at a constant rate) use
scripts/recording_alien_forager.py instead; it reuses this script's recording code.

1. Opens the webcam and shows a preview to check framing (SPACE = launch the task,
   ESC = cancel).
2. Creates the session folder raw_data/session_YYYYMMDD_HHMMSS_<task>/ and launches
   task/<task>.py --out-dir <session folder>.
3. Recording starts when the task prints "FACEPIPE_EVENT ready" on stdout (its start
   screen is up), so the video includes a baseline before the participant presses SPACE.
4. Recording stops when the task process exits (ESC in the task). The session counts as
   completed if the task printed "FACEPIPE_EVENT task_end" before exiting.

Session folder contents:

    recording_YYYY-MM-DD_HH-MM-SS.avi             the video
    recording_YYYY-MM-DD_HH-MM-SS_frametimes.csv  frame_idx, wall-clock time of each frame
                                                  (taken right after cap.read())
    task_events.csv                               timestamp, frame_idx, event, side; one row
                                                  per event (trial, reward, fail, switch,
                                                  system_flip). Written by the task as it
                                                  runs; this script fills in frame_idx
                                                  (nearest video frame) at the end.
    session_info.txt                              timing, camera, task details, event counts

Parse folder names with name.split("_", 3) -> ["session", date, time, task].
All times are the same PC wall clock (datetime.now() in both processes).

A task works with this script if it accepts --out-dir, writes task_events.csv there in
the format above, and prints FACEPIPE_EVENT lines (at least "ready" and "task_end").

    conda activate facepipe
    python scripts\\recording.py --task space_shooter --participant P01
    python scripts\\recording.py --task space_shooter --participant P01 --notes "glasses off"
"""
import argparse
import bisect
import csv
import hashlib
import json
import os
import queue
import subprocess
import sys
import threading
import time
from collections import Counter
from datetime import datetime

# --- Environment check: must run inside the facepipe conda env ---------------
try:
    import cv2
    import mediapipe  # noqa: F401  (not used here; confirms the right env)
except ImportError as e:
    print("\nERROR: could not import a required package ({}).".format(e.name))
    print("This script must run inside the 'facepipe' conda environment.")
    print("Open the Anaconda Prompt and run:")
    print("    conda activate facepipe")
    print("then run this script again.\n")
    sys.exit(1)

# --- Settings -----------------------------------------------------------------
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(PROJECT_DIR, "raw_data")
TASK_DIR = os.path.join(PROJECT_DIR, "task")

# The laptop's Integrated Camera is disabled in Device Manager, so the Rokoko
# Headcam is index 0. If the integrated camera is re-enabled the order can change;
# the resolution check in open_camera() warns about that.
CAMERA_INDICES = [0]
# Rokoko Headcam's largest mode (portrait); measured 59.9 fps sustained, and XVID
# encodes it at ~88 fps on this PC. Other modes: 960x1280 and 768x1024, also 60 fps.
REQUEST_WIDTH, REQUEST_HEIGHT = 1200, 1600
REQUEST_FPS = 60
MAX_DURATION_S = 20 * 60         # safety cap
PROGRESS_EVERY_S = 10
MAX_CONSECUTIVE_FAILS = 100      # ~2 s of failed reads -> camera is gone, stop
GAP_FACTOR = 1.5                 # frame interval > 1.5x expected counts as a gap
EVENT_PREFIX = "FACEPIPE_EVENT"
STDOUT_DRAIN_TIMEOUT_S = 2.0     # after the task exits, wait this long for its last stdout lines

WINDOW = "facepipe framing check (SPACE = launch task, ESC = cancel)"
KEY_ESC = 27
EVENTS_FILE = "task_events.csv"
EVENT_TYPES = ("trial", "reward", "fail", "switch", "system_flip")
# "ready" fields that describe the run rather than the task parameters
READY_NON_PARAMS = ("t", "session_id", "data_path", "start_side", "start_active")


def open_camera():
    """Try each camera index; return (cap, index) for the first that delivers a frame."""
    for index in CAMERA_INDICES:
        print("Trying camera index {} ...".format(index))
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, REQUEST_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, REQUEST_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, REQUEST_FPS)
        ok, frame = cap.read()   # some indices "open" but never deliver frames
        if ok:
            h, w = frame.shape[:2]
            if (w, h) != (REQUEST_WIDTH, REQUEST_HEIGHT):
                print("WARNING: camera {} gives {}x{}, not the Rokoko Headcam's max {}x{}. "
                      "Is this the right camera?".format(index, w, h, REQUEST_WIDTH, REQUEST_HEIGHT))
            return cap, index
        cap.release()
    return None, None


def read_task_output(stream, events):
    """Thread: parse the task's stdout; FACEPIPE_EVENT lines go to the queue, the rest is echoed."""
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
            print("  [task] " + line)
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
        cv2.putText(preview, "Centre the face, then press SPACE to launch the task",
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


def nearest_frame(frame_times, t):
    """Index of the frame closest in time to t, or "" if t is outside the video."""
    if t < frame_times[0] or t > frame_times[-1]:
        return ""
    i = bisect.bisect_left(frame_times, t)
    if i == 0:
        return 0
    return i if frame_times[i] - t < t - frame_times[i - 1] else i - 1


def add_frame_idx(events_path, frame_times):
    """Fill frame_idx in the task's event CSV in place; return its rows."""
    with open(events_path, newline="") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames
        rows = list(reader)
    for row in rows:
        row["frame_idx"] = nearest_frame(frame_times, datetime.fromisoformat(row["timestamp"]))
    with open(events_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)
    return rows


def default_progress(counts, task_info, elapsed):
    return "trials {} | rewards {} | switches {}".format(
        counts["trial"], counts["reward"], counts["switch"])


def run_session(task_name, participant="", notes="", progress=default_progress, extra_info=None):
    """Framing check, launch task/<task_name>.py, record, write the session folder.

    progress(counts, task_info, elapsed_s) -> str: task-specific part of the progress line
        (counts = Counter of event names seen so far, task_info = the "ready" fields).
    extra_info(rows, task_info) -> list of str: task-specific lines for session_info.txt
        (rows = task_events.csv rows with frame_idx filled in).
    """
    task_script = os.path.join(TASK_DIR, task_name + ".py")
    if not os.path.isfile(task_script):
        available = sorted(f[:-3] for f in os.listdir(TASK_DIR) if f.endswith(".py"))
        sys.exit("ERROR: task '{}' not found: {} does not exist.\nTasks in {}: {}".format(
            task_name, task_script, TASK_DIR, ", ".join(available) or "none"))

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
    # The writer needs an fps up front; fall back if the camera reports nonsense.
    writer_fps = reported_fps if 1 <= reported_fps <= 120 else REQUEST_FPS
    print("\nCamera {} opened: {}x{} @ {:.2f} fps (requested {}x{} @ {})".format(
        cam_index, width, height, reported_fps, REQUEST_WIDTH, REQUEST_HEIGHT, REQUEST_FPS))
    print("Check framing in the preview window, then press SPACE to launch {} (ESC = cancel).".format(task_name))

    if not framing_check(cap):
        cap.release()
        cv2.destroyAllWindows()
        print("\nCancelled. Nothing recorded.")
        return
    cv2.destroyAllWindows()
    cv2.waitKey(1)

    # --- Step 2: session folder + launch the task -------------------------------------
    launch_dt = datetime.now()
    session_name = "session_{}_{}".format(launch_dt.strftime("%Y%m%d_%H%M%S"), task_name)
    session_dir = os.path.join(RAW_DATA_DIR, session_name)
    os.makedirs(session_dir, exist_ok=False)

    env = dict(os.environ, PYGAME_HIDE_SUPPORT_PROMPT="1")
    task = subprocess.Popen([sys.executable, "-u", task_script, "--out-dir", session_dir],
                            cwd=TASK_DIR, stdout=subprocess.PIPE, stderr=None, text=True,
                            bufsize=1, env=env)
    events = queue.Queue()
    threading.Thread(target=read_task_output, args=(task.stdout, events), daemon=True).start()
    print("\nTask launched -> raw_data\\{}\\. Recording starts when the task is ready.".format(session_name))

    # --- Step 3: capture loop -----------------------------------------------------------
    counts = Counter()         # event name -> number seen on stdout
    first = {}                 # event name -> time of its first occurrence
    task_info = {}
    recording = False
    status = None              # completed / aborted / ...
    frames_q = writer = frametimes_file = writer_thread = None
    video_stem = None
    frame_times = []           # datetime of every written frame
    t_mono = []                # time.time() of every written frame, for fps / gaps
    read_fails = consecutive_fails = 0
    next_progress = PROGRESS_EVERY_S
    task_ended = False
    stdout_closed = False
    task_exit_time = None

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

            # Task events
            while not events.empty():
                name, fields = events.get()
                if name == "_stdout_closed":
                    stdout_closed = True
                    continue
                counts[name] += 1
                first.setdefault(name, fields.get("t"))
                if name == "ready" and not recording:
                    task_info = fields
                    # Video file name comes from the time recording starts.
                    video_stem = "recording_" + stamp.strftime("%Y-%m-%d_%H-%M-%S")
                    video_path = os.path.join(session_dir, video_stem + ".avi")
                    writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"XVID"),
                                             writer_fps, (width, height))
                    if not writer.isOpened():
                        print("ERROR: could not create video file: {}".format(video_path))
                        status = "could not create video file"
                        raise SystemExit
                    frametimes_file = open(os.path.join(session_dir, video_stem + "_frametimes.csv"), "w")
                    frametimes_file.write("frame_idx,wall_clock_timestamp\n")
                    frames_q = queue.Queue()
                    writer_thread = threading.Thread(target=write_frames,
                                                     args=(frames_q, writer, frametimes_file))
                    writer_thread.start()
                    recording = True
                    print("Task session {} ready -> recording started.".format(fields.get("session_id")))
                elif name == "task_start":
                    print("Task started.")
                elif name == "task_end":
                    task_ended = True
                    print("Task finished ({}). Recording until the task window is closed "
                          "(ESC in the task).".format(progress(counts, task_info, now - t_mono[0])
                                                      if t_mono else "no frames"))

            # Stop once the task has exited AND its stdout is fully read, so the last
            # events (e.g. "quit") aren't lost. The timeout covers a pipe that never
            # closes (e.g. held open by a child process of the task).
            if task_exit_time is None and task.poll() is not None:
                task_exit_time = now
            if task_exit_time is not None and events.empty() and (
                    stdout_closed or now - task_exit_time > STDOUT_DRAIN_TIMEOUT_S):
                if task_ended:
                    status = "completed"
                elif recording:
                    status = "aborted: task exited (code {}) before the end".format(task.returncode)
                else:
                    status = "task exited (code {}) before it was ready".format(task.returncode)
                break

            if recording and t_mono:
                elapsed = now - t_mono[0]
                if elapsed >= next_progress:
                    print("  recording {:4d} s | {} frames | {}".format(
                        int(elapsed), len(frame_times), progress(counts, task_info, elapsed)))
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

    if task.poll() is None:
        task.terminate()   # Ctrl+C / camera failure / safety cap: don't leave the task running
        task.wait()

    if not recording or not frame_times:
        print("\n{}. Nothing recorded.".format(status or "The task never became ready"))
        if not os.listdir(session_dir):
            os.rmdir(session_dir)
        else:
            print("Session folder kept (it has files): {}".format(session_dir))
        return

    # --- Step 4: event CSV (written by the task) -> add video frames ----------------------
    events_path = os.path.join(session_dir, EVENTS_FILE)
    if os.path.isfile(events_path):
        rows = add_frame_idx(events_path, frame_times)
        row_counts = Counter(r["event"] for r in rows)
        csv_status = "{} events ({}), frame_idx added".format(
            len(rows), ", ".join("{} {}".format(row_counts[e], e) for e in EVENT_TYPES))
    else:
        rows, row_counts = [], Counter()
        csv_status = "NOT WRITTEN by the task"
        print("WARNING: the task did not write {}".format(events_path))

    # --- Step 5: session info --------------------------------------------------------------
    n_frames = len(frame_times)
    duration = t_mono[-1] - t_mono[0]
    actual_fps = (n_frames - 1) / duration if duration > 0 else 0.0
    intervals = [b - a for a, b in zip(t_mono, t_mono[1:])]
    timing_gaps = sum(dt > GAP_FACTOR / writer_fps for dt in intervals)
    start_dt = frame_times[0]
    params = " ".join("{}={}".format(k, v) for k, v in task_info.items() if k not in READY_NON_PARAMS)
    info_lines = [
        "date: " + start_dt.strftime("%Y-%m-%d"),
        "time: " + start_dt.strftime("%H-%M-%S"),
        "duration_seconds: {:.2f}".format(duration),
        "total_frames: {}".format(n_frames),
        "actual_fps: {:.3f}".format(actual_fps),
        "actual_resolution: {}x{}".format(width, height),
        "camera_index: {}".format(cam_index),
        "participant: " + participant,
        "notes: " + notes,
        "",
        "# task",
        "task: {} ({})".format(task_name, os.path.relpath(task_script, PROJECT_DIR)),
        "task_script_sha256: " + file_sha256(task_script),
        "task_session_id: {}".format(task_info.get("session_id")),
        "task_status: " + status,
        "task_ready: {}".format(first.get("ready")),
        "task_start: {}".format(first.get("task_start", "not reached")),
        "task_end: {}".format(first.get("task_end", "not reached")),
        "task_parameters: " + params,
        "events: " + " ".join("{}={}".format(e, row_counts[e]) for e in EVENT_TYPES),
    ]
    if extra_info is not None:
        info_lines += extra_info(rows, task_info)
    info_lines += [
        "",
        "# technical",
        "first_frame: " + frame_times[0].isoformat(timespec="microseconds"),
        "last_frame: " + frame_times[-1].isoformat(timespec="microseconds"),
        "recording_start: task 'ready' event",
        "video_container_fps: {:.3f}".format(writer_fps),
        "camera_reported_fps: {:.3f}".format(reported_fps),
        "failed_frame_reads: {}".format(read_fails),
        "frame_timing_gaps: {}".format(timing_gaps),
        "stop_reason: " + status,
    ]
    with open(os.path.join(session_dir, "session_info.txt"), "w") as f:
        f.write("\n".join(info_lines) + "\n")

    # --- Step 6: summary --------------------------------------------------------------------------
    rel_session = os.path.relpath(session_dir, PROJECT_DIR)
    print("\nSession saved to: {}\\  ({})".format(rel_session, status))
    print("  {}.avi - {} frames, {:.1f} s, {:.2f} fps".format(video_stem, n_frames, duration, actual_fps))
    print("  {}_frametimes.csv - {} frame timestamps".format(video_stem, n_frames))
    print("  {} - {}".format(EVENTS_FILE, csv_status))
    print("  session_info.txt - written")
    if read_fails or timing_gaps:
        print("  note: {} failed reads, {} timing gaps (logged in session_info.txt)".format(
            read_fails, timing_gaps))
    if abs(actual_fps - writer_fps) > 1:
        print("  note: measured fps ({:.2f}) differs from the video file's fps ({:.2f}); "
              "use actual_fps / the frametimes file for timing.".format(actual_fps, writer_fps))
    print("\nNext step:")
    print("  python scripts\\st1_preprocess_face.py --session {}".format(rel_session))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", required=True,
                        help="task to run: the stem of a script in task/, e.g. space_shooter")
    parser.add_argument("--participant", default="", help="participant ID (e.g. P01), saved in session_info.txt")
    parser.add_argument("--notes", default="", help="free-text notes, saved in session_info.txt")
    args = parser.parse_args()
    if args.task == "alien_energy_forager":
        sys.exit("alien_energy_forager has its own recorder: "
                 "python scripts\\recording_alien_forager.py --participant <id>")
    run_session(args.task, args.participant, args.notes)


if __name__ == "__main__":
    main()
