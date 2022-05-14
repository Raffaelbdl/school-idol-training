import cv2
import subprocess
import time

import mediapipe as mp
from mediapipe.python.solutions import (
    drawing_utils as mp_drawing,
    drawing_styles as mp_drawing_styles,
    pose as mp_pose,
)
import numpy as np

from sip import Choregraphy
from sip.video_tools import test_camera


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


def dance_along(chore: Choregraphy) -> None:

    fps = test_camera()

    time.sleep(3.0)

    cap_trainee = cv2.VideoCapture(0)
    f_height = int(cap_trainee.get(cv2.CAP_PROP_FRAME_HEIGHT))
    f_width = int(cap_trainee.get(cv2.CAP_PROP_FRAME_WIDTH))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    cap_trainee.set(cv2.CAP_PROP_FPS, fps)
    out = cv2.VideoWriter("tmp.mp4", fourcc, fps, (f_width, f_height))

    p = subprocess.Popen(f"ffplay {chore.video_path} -autoexit", shell=True)
    while p.poll() is None:

        ret_trainee, trainee_frame = cap_trainee.read()
        if ret_trainee:
            out.write(trainee_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap_trainee.release()
    out.release()
    cv2.destroyAllWindows()
