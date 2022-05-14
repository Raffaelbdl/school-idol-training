import subprocess
import time

import cv2


def play_video_with_sound(video_path: str) -> None:
    """Uses ffmpeg"""
    p = subprocess.Popen(
        f"ffplay {video_path} -autoexit -hide_banner -loglevel error", shell=True
    )
    return p


def test_camera() -> int:
    """Returns fps"""
    cap = cv2.VideoCapture(0)

    start = time.time()
    n_frames = 0
    while cap.isOpened():
        n_frames += 1
        ret, frame = cap.read()
        cv2.imshow("Test Camera", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    total = time.time() - start

    cap.release()
    cv2.destroyAllWindows()

    return int(n_frames / total)
