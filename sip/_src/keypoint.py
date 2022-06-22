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
                
            frame_landmarks, pose_landmarks = capture_keypoints_from_frame(
                frame, pose, LANDMARK_NAMES, False, True
            )
            keypoints.append(frame_landmarks)
            landmarks.append(pose_landmarks)

    return keypoints, landmarks


def get_keypoints_from_stream(
    vid: cv2.VideoCapture, landmark_list: List[str]
) -> Dict[str, List[float]]:
    """Capture keypoints from a video stream

    Args:
        vid: a cv2 video stream
        landmark_list: the name of the joints to capture

    Outputs:
        frame_landmarks: the keypoints corresponding to the sampled frame
        pose_landmarks: the poses corresponding to the sampled frame (for
            eventual plotting)
    """

    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:

        ret, frame = vid.read()
        frame_landmarks, pose_landmarks = capture_keypoints_from_frame(
            frame, pose, landmark_list, True, True
        )
    return frame_landmarks, pose_landmarks


def capture_keypoints_from_frame(
    frame: np.ndarray,
    pose: mp_pose.Pose,
    landmark_list: List[str],
    camera_mirror: bool = False,
    vertical_mirror: bool = True,
):
    """Capture keypoints from a video frame

    Args:
        frame (np.ndarray): the output of cv2.VideoCapture().read()
        pose: the mp_pose in context
        landmark_list: the name of the joints to capture
        camera_mirror (bool): if True, keypoints are flipped horizontally
        vertical_mirror (bool): if True, keypoints are flipped vertically

    Outputs:
        frame_landmarks: the keypoints corresponding to the sampled frame
        pose_landmarks: the poses corresponding to the sampled frame (for
            eventual plotting)
    """

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame)
    pose_landmarks = results.pose_landmarks

    if pose_landmarks is not None:
        frame_landmarks = {}

        for idx, landmark in enumerate(pose_landmarks.landmark):

            landmark_name = landmark_list[idx]

            if landmark.HasField("visibility") and landmark.visibility >= 0.5:

                x_factor = -1 if camera_mirror else 1
                y_factor = -1 if vertical_mirror else 1
                frame_landmarks[landmark_name] = [
                    x_factor * landmark.x,  # ATTENTION MINUS X FOR MIRROR
                    y_factor * landmark.y,  # ATTENTION MINUS Y
                    landmark.z,
                ]
        return frame_landmarks, pose_landmarks
    else:
        return None, None


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
    return t_keypoints, t_visible


def split_keypoint(
    keypoint: List[Dict[str, List[float]]], n_splits: int
) -> List[List[Dict[str, List[float]]]]:
    """Returns n_splits keypoint lists"""
    split_keypoints = []
    split_length = len(keypoint) // n_splits

    for i in range(n_splits):
        split_keypoints.append(keypoint[i * split_length : (i + 1) * split_length])

    return split_keypoints
