"""Record a facepipe test session from a USB webcam.

Shows a live preview, records face video to a timestamped session folder in
raw_video/, writes session_info.txt and (optionally) copies the task's trial
CSV in as task_events.csv, so the folder is ready for the pipeline.

    recording_YYYY-MM-DD_HH-MM-SS.avi             the video
    recording_YYYY-MM-DD_HH-MM-SS_frametimes.csv  frame_idx, wall-clock time of each
                                                  written frame (taken right after cap.read())

Keys (preview window must be focused):
    S   = start recording
    Q   = stop recording and save
    ESC = cancel without saving

Run inside the facepipe conda environment:
    python scripts\\record_session.py
"""
import os
import shutil
import sys
import time
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
RAW_VIDEO_DIR = os.path.join(PROJECT_DIR, "raw_video")

CAMERA_INDICES = [1, 0]          # external USB camera first, then built-in
REQUEST_WIDTH, REQUEST_HEIGHT = 1280, 720
REQUEST_FPS = 30
MAX_DURATION_S = 10 * 60         # safety cap
PROGRESS_EVERY_S = 5
MAX_CONSECUTIVE_FAILS = 100      # ~3 s of failed reads -> camera is gone, stop
GAP_FACTOR = 1.5                 # frame interval > 1.5x expected counts as a gap

WINDOW = "facepipe recording (S = start, Q = stop & save, ESC = cancel)"
KEY_ESC = 27


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
        ok, _ = cap.read()   # some indices "open" but never deliver frames
        if ok:
            return cap, index
        cap.release()
    return None, None


def ask_task_csv():
    raw = input("Enter path to task CSV file, or press Enter to skip: ").strip()
    path = raw.strip('"').strip("'")   # Windows "Copy as path" adds quotes
    if not path:
        return None
    if not os.path.isfile(path):
        print("WARNING: file not found right now: {}".format(path))
        print("         Will check again after recording (the task may create it).")
    return path


def draw_rec(frame, elapsed_s):
    """Draw a red REC indicator + elapsed time on a preview copy of the frame."""
    cv2.circle(frame, (30, 30), 10, (0, 0, 255), -1)
    label = "REC {:02d}:{:02d}".format(int(elapsed_s) // 60, int(elapsed_s) % 60)
    cv2.putText(frame, label, (48, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 0, 255), 2, cv2.LINE_AA)


def window_closed():
    try:
        return cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return True


def main():
    task_csv = ask_task_csv()

    # --- Step 1: camera setup ---------------------------------------------------
    cap, cam_index = open_camera()
    if cap is None:
        print("\nERROR: could not open a camera (tried indices {}).".format(CAMERA_INDICES))
        print("Check the USB webcam is plugged in and not in use by another app "
              "(Teams, Zoom, Camera app), then try again.\n")
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    reported_fps = cap.get(cv2.CAP_PROP_FPS)
    print("\nCamera {} opened.".format(cam_index))
    print("  requested: {}x{} @ {} fps".format(REQUEST_WIDTH, REQUEST_HEIGHT, REQUEST_FPS))
    print("  accepted:  {}x{} @ {:.2f} fps (as reported by the camera)".format(
        width, height, reported_fps))
    # The writer needs an fps up front; fall back if the camera reports nonsense.
    writer_fps = reported_fps if 1 <= reported_fps <= 120 else REQUEST_FPS

    print("\nCheck framing in the preview window, then:")
    print("  Press S to start recording")
    print("  Press Q to stop recording and save")
    print("  Press ESC to cancel without saving\n")

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, width, height)

    recording = False
    cancelled = False
    stop_reason = None
    writer = None
    frametimes_file = None
    session_dir = None
    start_dt = None
    frames = 0
    read_fails = 0
    consecutive_fails = 0
    timing_gaps = 0
    t_first = t_last = None
    next_progress = PROGRESS_EVERY_S
    expected_interval = 1.0 / writer_fps

    try:
        while True:
            ok, frame = cap.read()
            now = time.time()
            frame_wall_clock = datetime.now().isoformat(timespec="microseconds")

            if not ok:
                consecutive_fails += 1
                if recording:
                    read_fails += 1
                if consecutive_fails >= MAX_CONSECUTIVE_FAILS:
                    print("\nERROR: camera stopped delivering frames.")
                    stop_reason = "camera stopped delivering frames"
                    break
                if cv2.waitKey(1) & 0xFF == KEY_ESC:
                    cancelled = True
                    break
                continue
            consecutive_fails = 0

            if recording:
                if t_last is not None and now - t_last > GAP_FACTOR * expected_interval:
                    timing_gaps += 1
                if t_first is None:
                    t_first = now
                t_last = now
                writer.write(frame)
                frametimes_file.write("{},{}\n".format(frames, frame_wall_clock))
                frames += 1
                elapsed = now - t_first

                if elapsed >= next_progress:
                    print("  recording... {:4d} s  |  {} frames".format(int(elapsed), frames))
                    next_progress += PROGRESS_EVERY_S
                if elapsed >= MAX_DURATION_S:
                    print("\nReached the {}-minute safety cap, stopping.".format(
                        MAX_DURATION_S // 60))
                    stop_reason = "max duration reached"
                    break

                preview = frame.copy()   # keep the REC overlay out of the saved video
                draw_rec(preview, elapsed)
                cv2.imshow(WINDOW, preview)
            else:
                cv2.imshow(WINDOW, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("s"), ord("S")) and not recording:
                # --- Step 2: session folder -----------------------------------
                start_dt = datetime.now()
                session_name = "session_" + start_dt.strftime("%Y-%m-%d_%H-%M-%S")
                session_dir = os.path.join(RAW_VIDEO_DIR, session_name)
                os.makedirs(session_dir, exist_ok=False)
                # The pipeline reads the start time from this filename; the
                # _frametimes.csv sidecar gives the exact time of every frame.
                video_stem = "recording_" + start_dt.strftime("%Y-%m-%d_%H-%M-%S")
                video_name = video_stem + ".avi"
                frametimes_name = video_stem + "_frametimes.csv"
                video_path = os.path.join(session_dir, video_name)
                fourcc = cv2.VideoWriter_fourcc(*"XVID")
                writer = cv2.VideoWriter(video_path, fourcc, writer_fps, (width, height))
                if not writer.isOpened():
                    print("ERROR: could not create video file: {}".format(video_path))
                    writer = None
                    shutil.rmtree(session_dir, ignore_errors=True)
                    session_dir = None
                    break
                frametimes_file = open(os.path.join(session_dir, frametimes_name), "w")
                frametimes_file.write("frame_idx,wall_clock_timestamp\n")
                recording = True
                print("Recording started -> raw_video\\{}\\".format(session_name))
            elif key in (ord("q"), ord("Q")):
                stop_reason = "stopped by user"
                break
            elif key == KEY_ESC:
                cancelled = True
                break
            elif window_closed():
                # Closing the window counts as Q while recording, ESC otherwise.
                if recording:
                    stop_reason = "preview window closed"
                else:
                    cancelled = True
                break
    except KeyboardInterrupt:
        print("\nCtrl+C received, stopping and saving.")
        stop_reason = "Ctrl+C"
    finally:
        if writer is not None:
            writer.release()
        if frametimes_file is not None:
            frametimes_file.close()
        cap.release()
        cv2.destroyAllWindows()

    if not recording:
        print("\nNo recording made. Nothing saved.")
        return
    if cancelled:
        shutil.rmtree(session_dir, ignore_errors=True)
        print("\nRecording cancelled. Session folder deleted, nothing saved.")
        return
    if frames == 0:
        shutil.rmtree(session_dir, ignore_errors=True)
        print("\nNo frames were captured. Session folder deleted.")
        return

    # --- Step 4: session info -----------------------------------------------------
    duration = t_last - t_first
    actual_fps = (frames - 1) / duration if duration > 0 else 0.0
    info_lines = [
        "date: " + start_dt.strftime("%Y-%m-%d"),
        "time: " + start_dt.strftime("%H-%M-%S"),
        "duration_seconds: {:.2f}".format(duration),
        "total_frames: {}".format(frames),
        "actual_fps: {:.3f}".format(actual_fps),
        "actual_resolution: {}x{}".format(width, height),
        "camera_index: {}".format(cam_index),
        "notes: ",
        "",
        "# technical",
        "video_container_fps: {:.3f}".format(writer_fps),
        "camera_reported_fps: {:.3f}".format(reported_fps),
        "failed_frame_reads: {}".format(read_fails),
        "frame_timing_gaps: {}".format(timing_gaps),
        "stop_reason: {}".format(stop_reason),
    ]
    with open(os.path.join(session_dir, "session_info.txt"), "w") as f:
        f.write("\n".join(info_lines) + "\n")

    # --- Step 5: task CSV -----------------------------------------------------------
    if task_csv is None:
        csv_status = "not provided"
    elif os.path.isfile(task_csv):
        shutil.copy2(task_csv, os.path.join(session_dir, "task_events.csv"))
        csv_status = "copied"
    else:
        print("WARNING: task CSV not found, not copied: {}".format(task_csv))
        csv_status = "not found (path: {})".format(task_csv)

    # --- Step 6: summary --------------------------------------------------------------
    rel_session = os.path.relpath(session_dir, PROJECT_DIR)
    print("\nSession saved to: {}\\".format(rel_session))
    print("  {} — {} frames, {:.1f} seconds, {:.2f}fps".format(
        video_name, frames, duration, actual_fps))
    print("  {} — {} frame timestamps".format(frametimes_name, frames))
    print("  session_info.txt — written")
    print("  task_events.csv  — {}".format(csv_status))
    if read_fails or timing_gaps:
        print("  note: {} failed reads, {} timing gaps (logged in session_info.txt)".format(
            read_fails, timing_gaps))
    if abs(actual_fps - writer_fps) > 1:
        print("  note: measured fps ({:.2f}) differs from the video file's fps ({:.2f}); "
              "use actual_fps for timing.".format(actual_fps, writer_fps))
    print("\nReady for pipeline. Next step:")
    print("  python scripts\\mediapipe_landmarks.py --session {}".format(rel_session))


if __name__ == "__main__":
    main()
