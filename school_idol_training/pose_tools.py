import enum
from typing import Dict, List

import cv2
import numpy as np
from scipy.interpolate import interp1d

from school_idol_training import Choregraphy


LANDMARK_NAMES = [
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky_1",
    "right_pinky_1",
    "left_index_1",
    "right_index_1",
    "left_thumb_2",
    "right_thumb_2",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
]


def extrapolate_keypoints(
    chore: Choregraphy, new_fps: int
) -> List[Dict[str, List[float]]]:
    cap = cv2.VideoCapture(chore.video_path)
    cur_fps = int(cap.get(cv2.CAP_PROP_FPS))

    cur_t = np.linspace(0, 1, 1 / len(chore.keypoints))
    new_t = np.linspace(0, 1, 1 / (len(chore.keypoints) * new_fps / cur_fps))

    new_keypoints = []
    for keypoint in chore.keypoints:
        if keypoint is None:
            new_keypoints.append(None)
        else:
            new_keypoint = {}
            for name in new_keypoint.keys():
                named_keypoint = []
                for k in new_keypoint[name]:
                    pass


def keypoints_as_time_series(keypoints: List[Dict[str, List[float]]]):
    """Makes time series of keypoints

    Args:
        keypoints (List[Dict[str, List[float]]])
    Returns:
        time_keypoints (Dict[str, List[List[float]]])
            where keys are joint names
            and values are sequences of list for coordinates
    """
    t_keypoints = {name: [] for name in LANDMARK_NAMES}
    t_visible = {name: [] for name in LANDMARK_NAMES}

    for i in range(len(keypoints)):
        if keypoints[i] is not None:
            not_none = i
            break
        else:
            for name in t_keypoints.keys():
                t_keypoints[name].append([-10, -10])

    for f in range(not_none, len(keypoints)):
        f_keypoint = keypoints[f]
        for name in t_keypoints.keys():
            try:
                t_keypoints[name].append(f_keypoint[name][:2])
                t_visible[name].append(f)
            except KeyError or TypeError:
                t_keypoints[name].append([-10, -10])

    return t_keypoints, t_visible


def fill_time_serie(sequence: np.ndarray, t_visible: List[int], fps: int) -> np.ndarray:

    if len(t_visible) > 0:
        # initialize two first points
        if t_visible[1] != t_visible[0] + 1:
            sequence[t_visible[1]] = sequence[t_visible[0]]

        for t in range(t_visible[0] + 2, t_visible[-1]):
            v = (sequence[t - 2] + sequence[t - 1]) / 2
            if np.all(np.equal(sequence[t], np.array([-10, -10]))):
                sequence[t] = v / fps + sequence[t - 1]

    return sequence
