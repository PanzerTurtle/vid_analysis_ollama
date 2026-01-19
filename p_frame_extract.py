import os
import cv2
import json
import shutil


def extract_frames(video_path, out_dir, interval_sec=5,  use_motion=True):
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval_sec)

    prev_frame = None
    frame_id = saved = skipped = 0
    timestamps = {}

    print(f"Video: {video_path}")
    print(f"FPS: {fps:.2f}, Interval: {interval_sec}s")
    print(f"Mode: {'Motion detection' if use_motion else 'All frames'}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % frame_interval == 0:
            save = True
            timestamp = frame_id / fps

            if use_motion and prev_frame is not None:
                save = detect_motion(prev_frame, frame)
                if not save:
                    skipped += 1

            if save:
                saved += 1
                name = f"frame_{saved}.jpg"
                path = os.path.join(out_dir, name)
                cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                timestamps[name] = timestamp

            if use_motion:
                prev_frame = frame.copy()

        frame_id += 1

    cap.release()

    with open(os.path.join(out_dir, "timestamps.json"), "w") as f:
        json.dump(timestamps, f, indent=2)

    print(f"Saved: {saved}, Skipped: {skipped}")
    return timestamps



def detect_motion(frame1, frame2, threshold=8000) -> bool:
    gray1 = cv2.GaussianBlur(
        cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY), (21, 21), 0
    )
    gray2 = cv2.GaussianBlur(
        cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY), (21, 21), 0
    )

    diff = cv2.absdiff(gray1, gray2)
    _, thresh_img = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

    return cv2.countNonZero(thresh_img) > threshold
