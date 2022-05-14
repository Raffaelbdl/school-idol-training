import time
from typing import Any, Dict, List, Tuple

from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmarkList
from mediapipe.python.solutions import (
    drawing_utils as mp_drawing,
    drawing_styles as mp_drawing_styles,
    pose as mp_pose,
)
import numpy as np
import cv2
from tqdm import tqdm

from school_idol_training import Choregraphy


def display_chore(chore: Choregraphy, flip: bool = False) -> None:
    """Displays a chore"""

    cap = cv2.VideoCapture(chore.video_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    sleep = 1 / fps

    for i in range(n_frames):
        t1 = time.time()
        ret, annotated_frame = cap.read()
        # mp_drawing.draw_landmarks(
        #     annotated_frame,
        #     chore.landmarks[i],
        #     mp_pose.POSE_CONNECTIONS,
        #     landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
        # )
        if flip:
            annotated_frame = cv2.flip(annotated_frame, 1)
        cv2.imshow(chore.title, annotated_frame)
        t1 = time.time() - t1
        waitfor = max(1, int((sleep - t1) * 1000))
        print(waitfor)
        if cv2.waitKey(waitfor) & 0xFF == ord("q"):
            break
