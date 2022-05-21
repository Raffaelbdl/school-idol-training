from typing import Dict, List, Optional, Tuple

import cv2
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmarkList
from mediapipe.python.solutions import pose as mp_pose
import numpy as np
from tqdm import tqdm

from sip._src.metadata import LANDMARK_NAMES


def get_keypoints_from_video_file(
    filepath: str, load_message: Optional[str] = None
) -> Tuple[List[Dict[str, List[float]]], NormalizedLandmarkList]:
    """Gets keypoints from array for each frame

    Returns:
        keypoints (list): List of dictionaries where keys are keypoints' names and
            values are a list of coordinates of each keypoint
        landmarks (list): List of mediapipe landmarks for each frame (for plot)
    """
    keypoints = []
    landmarks = []
    cap = cv2.VideoCapture(filepath)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        iterator = (
            tqdm(range(n_frames), desc=load_message)
            if load_message is not None
            else tqdm(range(n_frames), desc="Making prediction from array ...")
        )
        for i in iterator:

            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame)
            pose_landmarks = results.pose_landmarks

            if pose_landmarks is not None:
                frame_landmarks = {}
                for idx, landmark in enumerate(pose_landmarks.landmark):
                    landmark_name = LANDMARK_NAMES[idx]
                    if landmark.HasField("visibility") and landmark.visibility >= 0.5:
                        frame_landmarks[landmark_name] = [
                            landmark.x,
                            landmark.y,
                            landmark.z,
                        ]
                keypoints.append(frame_landmarks)
                landmarks.append(pose_landmarks)
            else:
                keypoints.append(None)
                landmarks.append(None)

    return keypoints, landmarks


def keypoints_to_time_series(
    keypoints: List[Dict[str, List[float]]]
) -> Tuple[Dict[str, np.ndarray]]:
    """Transforms keypoints to time series

    * If a keypoint is None (does not exist, its corresponding value
    will be [-10, -10])

    * Only x and y dimensions are used !

    Args:
        keypoints (List[Dict[str, List[float]]])
    Returns:
        time_keypoints (Dict[str, np.ndarray])
            where keys are joint names
            and values are sequences of list for coordinates
        time_visibles (Dict[str, np.ndarray])
            where keys are joint names
            and values are booleans to tell if joint is visible
    """
    t_keypoints = {name: [] for name in LANDMARK_NAMES}
    t_visible = {name: [] for name in LANDMARK_NAMES}

    for t in range(len(keypoints)):
        if keypoints[t] is None:
            for name in t_keypoints.keys():
                t_keypoints[name].append([-10, -10])
                t_visible[name].append(0)
        else:
            for name in t_keypoints.keys():
                try:
                    t_keypoints[name].append(keypoints[t][name][:2])
                    t_visible[name].append(1)
                except KeyError:
                    t_keypoints[name].append([-10, -10])
                    t_visible[name].append(0)

    t_keypoints = {k: np.array(v) for (k, v) in t_keypoints.items()}
    t_visible = {k: np.array(v) for (k, v) in t_visible.items()}

    return add_neck_and_hip(t_keypoints, t_visible)


def add_neck_and_hip(
    t_keypoints: Dict[str, np.ndarray], t_visible: Dict[str, np.ndarray]
) -> Tuple[Dict[str, np.ndarray]]:
    t_keypoints["neck"] = (
        t_keypoints["right_shoulder"] + t_keypoints["left_shoulder"]
    ) / 2
    t_visible["neck"] = t_visible["right_shoulder"] * t_visible["left_shoulder"]
    t_keypoints["middle_hip"] = (t_keypoints["right_hip"] + t_keypoints["left_hip"]) / 2
    t_visible["middle_hip"] = t_visible["right_hip"] * t_visible["left_hip"]

    return t_keypoints, t_visible
