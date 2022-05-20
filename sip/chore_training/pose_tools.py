from typing import Dict, List

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

from sip import Choregraphy, LANDMARK_NAMES, JOINT_PAIRS


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

    not_none = 0
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
            except (KeyError, TypeError):
                t_keypoints[name].append([-10, -10])

    return t_keypoints, t_visible


def fill_time_serie(sequence: np.ndarray, t_visible: List[int], fps: int) -> np.ndarray:

    if len(t_visible) > 1:
        # initialize two first points
        if t_visible[1] != t_visible[0] + 1:
            sequence[t_visible[1]] = sequence[t_visible[0]]

        for t in range(t_visible[0] + 2, t_visible[-1]):
            v = sequence[t - 2] + sequence[t - 1]
            if np.all(np.equal(sequence[t], np.array([-10, -10]))):
                sequence[t] = np.clip(v / fps + sequence[t - 1], 0, 1)

    return sequence


def extrapolate_time_serie(sequence: np.ndarray, new_length: int) -> np.ndarray:

    cur_t = np.linspace(0, 1, len(sequence))
    new_t = np.linspace(0, 1, new_length)

    new_sequence = np.empty(shape=(new_length, *sequence.shape[1:]))
    for i in range(new_sequence.shape[-1]):
        min_val = np.min(sequence[..., i] > 10)
        interpoli = interp1d(cur_t, sequence[..., i])
        new_val = interpoli(new_t)
        new_val = np.where(new_val >= min_val, new_val, -10)
        new_sequence[..., i] = new_val
    return new_sequence


def plot_chore(chore: Choregraphy, keypoint_name: str) -> None:
    cap = cv2.VideoCapture(chore.video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    t_keypoints, t_visible = keypoints_as_time_series(chore.keypoints)

    new_keypoints = {
        name: extrapolate_time_serie(
            fill_time_serie(np.array(sequence), t_visible[name], fps), 500
        )
        for name, sequence in t_keypoints.items()
    }
    plt.plot(new_keypoints[keypoint_name])
    plt.set_ylim(bottom=0)
    plt.legend(["x", "y"])
    plt.show()


def cosine_similarity(
    sequenceA: np.ndarray,
    sequenceB: np.ndarray,
    sequenceC: np.ndarray,
    sequenceD: np.ndarray,
):
    """Computes the cosine similarity between AB and CD"""
    AB = sequenceB - sequenceA
    CD = sequenceD - sequenceC
    normAB = np.sqrt(np.sum(np.square(AB), axis=-1))
    normCD = np.sqrt(np.sum(np.square(CD), axis=-1))
    return np.sum(AB * CD, axis=-1) / (normAB * normCD + 1e-3)


def get_cosine_similarity(t_keypoints1, t_keypoints2, joint1: str, joint2: str):
    sequenceA = np.array(t_keypoints1[joint1])
    sequenceB = np.array(t_keypoints1[joint2])
    sequenceC = np.array(t_keypoints2[joint1])
    sequenceD = np.array(t_keypoints2[joint2])
    return cosine_similarity(sequenceA, sequenceB, sequenceC, sequenceD)


def get_worst_cosine_score(t_keypoints):
    scores = []
    for joint_pair in JOINT_PAIRS:
        scores.append(-1 * np.ones_like(t_keypoints[joint_pair[0]]))
    return np.sum(scores)


def get_best_cosine_score(t_keypoints):
    scores = []
    for joint_pair in JOINT_PAIRS:
        scores.append(1 * np.ones_like(t_keypoints[joint_pair[0]]))
    return np.sum(scores)


def get_cosine_score(t_keypoints1, t_keypoints2):
    scores = []
    for joint_pair in JOINT_PAIRS:
        scores.append(
            get_cosine_similarity(
                t_keypoints1, t_keypoints2, joint_pair[0], joint_pair[1]
            )
        )
        print(joint_pair, scores[-1])
    return np.sum(scores)
