from dataclasses import dataclass, field
from typing import Dict, List

from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmarkList
import numpy as np


JOINT_PAIRS = [
    ["right_elbow", "right_wrist"],
    ["right_shoulder", "right_elbow"],
    ["right_shoulder", "right_hip"],
    ["right_hip", "right_knee"],
    ["right_knee", "right_ankle"],
    ["left_elbow", "left_wrist"],
    ["left_shoulder", "left_elbow"],
    ["left_shoulder", "left_hip"],
    ["left_hip", "left_knee"],
    ["left_knee", "left_ankle"],
]

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


@dataclass
class Choregraphy:
    """A choregraphy is described by its title and its video"""

    title: str
    keypoints: List[Dict[str, List[float]]]
    landmarks: List[NormalizedLandmarkList]

    video_path: str = field(default=None)
    baseline_keypoints: Dict[str, np.ndarray] = field(default=None)
    baseline_score: float = field(default=None)

    score: float = field(default=None)  # for trainee
