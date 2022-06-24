from typing import Dict, List, Optional, Tuple, Union

import cv2
from mediapipe.python.solutions import pose as mp_pose
import numpy as np
from tqdm import tqdm

from sip._src import Keypoints, Landmarks, LANDMARK_NAMES


def get_keypoints_from_video_file(
    filepath: str, load_message: Optional[str] = None
) -> Tuple[List[Union[Keypoints, None]], List[Union[Landmarks, None]]]:
    """Get keypoints from a video file

    Args:
        filepath (str): path to video, eg. "path/to/video.mp4"
        load_message (str): message to display when compiling the chore

    Returns:
        keypoints_list: List of Keypoints
        landmarks_list: List of Landmarks
    """
    keypoints_list, landmarks_list = [], []

    cap = cv2.VideoCapture(filepath)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if load_message is not None:
        iterator = tqdm(range(n_frames), desc=load_message)
    else:
        iterator = tqdm(range(n_frames), desc="Compiling choregraphy ...")

    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:

        for _ in iterator:

            ret, frame = cap.read()
            if not ret:
                keypoints_list.append(None)
                keypoints_list.append(None)
                continue

            frame_keypoints, frame_landmarks = get_keypoints_from_frame(
                frame, pose, LANDMARK_NAMES, False, True
            )
            keypoints_list.append(frame_keypoints)
            landmarks_list.append(frame_landmarks)

    return keypoints_list, landmarks_list


def get_keypoints_from_stream(
    vid: cv2.VideoCapture, landmark_names: List[str]
) -> Union[Tuple[Keypoints, Landmarks], Tuple[None, None]]:
    """Get keypoints from a video stream

    Args:
        vid: a cv2 video stream
        landmark_names (List[str]): the names of the joints to capture ordered

    Returns:
        frame_keypoints: the keypoints corresponding to the sampled frame
        frame_landmarks: the poses corresponding to the sampled frame (for
            eventual plotting)
    """
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:

        ret, frame = vid.read()
        if not ret:
            return None, None

        frame_keypoints, frame_landmarks = get_keypoints_from_frame(
            frame, pose, landmark_names, True, True
        )

    return frame_keypoints, frame_landmarks


def get_keypoints_from_frame(
    frame: np.ndarray,
    pose: mp_pose.Pose,
    landmark_names: List[str],
    camera_mirror: bool = False,
    vertical_mirror: bool = True,
) -> Union[Tuple[Keypoints, Landmarks], Tuple[None, None]]:
    """Capture keypoints from a video frame

    Args:
        frame (np.ndarray): the output of cv2.VideoCapture().read()
        pose (mp_pose.Pose): the mp_pose in context
        landmark_names (List[str]): the name of the joints to capture
        camera_mirror (bool): if True, keypoints are flipped horizontally
        vertical_mirror (bool): if True, keypoints are flipped vertically

    Returns:
        frame_keypoints: the keypoints corresponding to the sampled frame
        frame_landmarks: the landmarks corresponding to the sampled frame (for
            eventual plotting)
    """
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame)
    frame_landmarks = results.pose_landmarks

    if frame_landmarks is not None:
        frame_keypoints = {}

        for idx, landmark in enumerate(frame_landmarks.landmark):

            landmark_name = landmark_names[idx]

            if landmark.HasField("visibility") and landmark.visibility >= 0.5:

                x_factor = -1 if camera_mirror else 1
                y_factor = -1 if vertical_mirror else 1
                frame_keypoints[landmark_name] = [
                    x_factor * landmark.x,
                    y_factor * landmark.y,
                    landmark.z,
                ]

        return frame_keypoints, frame_landmarks

    else:
        return None, None


def keypoints_to_time_series(
    keypoints_list: List[Keypoints],
) -> Tuple[Dict[str, np.ndarray]]:
    """Transform keypoints to time series

    * If a keypoint is None (does not exist, its corresponding value
    will be [-10, -10])

    * Only x and y dimensions are used !

    Args:
        keypoints_list (List[Keypoiints])

    Returns:
        time_keypoints (Dict[str, np.ndarray])
            where keys are landmarks names
            and values are sequences of list for coordinates
        time_visible (Dict[str, np.ndarray])
            where keys are landmarks names
            and values are booleans to tell if joint is visible
    """
    t_keypoints = {name: [] for name in LANDMARK_NAMES}
    t_visible = {name: [] for name in LANDMARK_NAMES}

    for t in range(len(keypoints_list)):

        for name in t_keypoints.keys():
            if keypoints_list[t] is None or name not in keypoints_list[t].keys():
                t_keypoints[name].append([-10, -10])
                t_visible[name].append(0)
            else:
                t_keypoints[name].append(keypoints_list[t][name][:2])
                t_visible[name].append(1)

    t_keypoints = {k: np.array(v) for (k, v) in t_keypoints.items()}
    t_visible = {k: np.array(v) for (k, v) in t_visible.items()}

    return t_keypoints, t_visible


def split_keypoint(
    keypoints_list: List[Keypoints], n_splits: int
) -> List[List[Dict[str, List[float]]]]:
    """Split a list of keypoints

    Args:
        keypoints_list (List[Keypoints])
        n_splits (int): number of segments

    Returns:
        split_keypoints_list (List[List[Keypoints]])
    """
    split_keypoints_list = []
    split_length = len(keypoints_list) // n_splits

    for i in range(n_splits):
        split_keypoints_list.append(
            keypoints_list[i * split_length : (i + 1) * split_length]
        )

    return split_keypoints_list
