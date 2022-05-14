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


def write_video(n_frames: int, fps: int) -> None:

    cap = cv2.VideoCapture(0)

    f_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    f_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter("tmp.mp4", fourcc, fps, (f_width, f_height))

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
