import json
import os
import pickle
import shutil
from typing import Any, Dict, List, Optional, Tuple

from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmarkList
from mediapipe.python.solutions import (
    drawing_utils as mp_drawing,
    drawing_styles as mp_drawing_styles,
    pose as mp_pose,
)
import numpy as np
import cv2
from tqdm import tqdm

from sip import Choregraphy
from sip.chore_training.scoring import (
    keypoints_as_time_series,
    find_baseline_sequence_dtw,
    get_dtw_from_keypoints,
)

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


def array_from_video_file(filepath: str) -> Tuple[np.ndarray, int]:
    """Creates an array from a video"""
    cap = cv2.VideoCapture(filepath)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    f_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    f_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    video_array = np.empty((n_frames, f_height, f_width, 3), dtype=np.uint8)

    for i in tqdm(range(n_frames), desc="Loading video in array ..."):
        ret, frame = cap.read()
        if not ret:
            break
        video_array[i] = frame
    cap.release()

    return video_array, fps


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


def make_chore_from_file(
    title: str, filepath: str, load_message: Optional[str] = None
) -> Choregraphy:
    """Creates a choregraphy from a video"""
    assert os.path.isfile(filepath)
    keypoints, landmarks = get_keypoints_from_video_file(
        filepath=filepath, load_message=load_message
    )
    baseline_keypoints, baseline_score = find_baseline_sequence_dtw(keypoints, filepath)
    return Choregraphy(
        title=title,
        keypoints=keypoints,
        landmarks=landmarks,
        video_path=filepath,
        baseline_keypoints=baseline_keypoints,
        baseline_score=baseline_score,
    )


def save_chore(chore: Choregraphy, dirpath: str) -> None:
    chore_path = os.path.join(dirpath, chore.title)
    if not os.path.isdir(chore_path):
        os.makedirs(chore_path)

    shutil.copy(chore.video_path, os.path.join(chore_path, "video.mp4"))

    with open(os.path.join(chore_path, "keypoints"), "wb") as f:
        pickle.dump(chore.keypoints, f)
    with open(os.path.join(chore_path, "landmarks"), "wb") as f:
        pickle.dump(chore.landmarks, f)
    with open(os.path.join(chore_path, "baseline_keypoints"), "wb") as f:
        pickle.dump(chore.baseline_keypoints, f)
    with open(os.path.join(chore_path, "score"), "wb") as f:
        pickle.dump(chore.score, f)


def load_chore(chore_path: str) -> Choregraphy:
    chore_path = chore_path.rstrip("/")
    title = chore_path.split("/")[-1]
    with open(os.path.join(chore_path, "keypoints"), "rb") as f:
        keypoints = pickle.load(f)
    with open(os.path.join(chore_path, "landmarks"), "rb") as f:
        landmarks = pickle.load(f)
    with open(os.path.join(chore_path, "baseline_keypoints"), "rb") as f:
        baseline_keypoints = pickle.load(f)
    with open(os.path.join(chore_path, "score"), "rb") as f:
        score = pickle.load(f)

    _, baseline_score = find_baseline_sequence_dtw(
        keypoints, os.path.join(chore_path, "video.mp4"), baseline_keypoints
    )

    return Choregraphy(
        title=title,
        keypoints=keypoints,
        landmarks=landmarks,
        video_path=os.path.join(chore_path, "video.mp4"),
        baseline_score=baseline_score,
        score=score,
    )
