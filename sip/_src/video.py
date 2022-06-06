"""Video related functions"""
import os
import subprocess
import time

import cv2
from mediapipe.python.solutions import (
    drawing_utils as mp_drawing,
    drawing_styles as mp_drawing_styles,
    pose as mp_pose,
)
import numpy as np


def play_video_with_sound(video_path: str) -> subprocess.Popen:
    """Uses ffmpeg to play a video

    Args:
        video_path (str): Path to video
    Returns:
        A subprocess that launch the video
    """
    p = subprocess.Popen(
        f"ffplay {video_path} -autoexit -hide_banner -loglevel error -fs", shell=True
    )
    return p


def test_camera(display_joints: bool = False) -> int:
    """Computes the camera framerate"""
    # necessary to make test_camera window at front
    _open_dummy_cv2_window()

    cap = cv2.VideoCapture(0)

    start = time.time()
    n_frames = 0
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        while cap.isOpened():
            _, frame = cap.read()

            if display_joints:
                frame.flags.writeable = False
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(frame)
                frame.flags.writeable = True
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
                )

            n_frames += 1
            cv2.imshow("Test Camera", cv2.flip(frame, 1))
            cv2.setWindowProperty("Test Camera", cv2.WND_PROP_TOPMOST, 1)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    duration = time.time() - start
    cap.release()
    cv2.destroyAllWindows()

    return int(n_frames / duration)


def record_camera(video_name: str, dir_path: str, n_frames: int) -> None:
    """Records the camera
    for a given number of frames
    """
    fps = test_camera(True)
    time.sleep(3.0)

    video_path = os.path.join(dir_path, video_name + ".mp4")
    cap = cv2.VideoCapture(0)

    f_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    f_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(video_path, fourcc, fps, (f_width, f_height))

    for i in range(n_frames):
        ret, frame = cap.read()

        if ret:
            out.write(frame)
            cv2.imshow("Capture", cv2.flip(frame, 1))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

    cap.release()
    out.release()

    cv2.destroyAllWindows()


def _open_dummy_cv2_window():
    cv2.namedWindow("dummy", cv2.WINDOW_NORMAL)
    img = np.ones((100, 100, 3))
    cv2.imshow("dummy", img)
    cv2.setWindowProperty("dummy", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.waitKey(1)
    cv2.setWindowProperty("dummy", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
    cv2.destroyWindow("dummy")


def get_duration(video_path: str) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            video_path,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return float(result.stdout)
