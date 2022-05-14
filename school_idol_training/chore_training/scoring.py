from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from fastdtw import fastdtw

from school_idol_training import Choregraphy
from school_idol_training.chore_training.pose_tools import (
    extrapolate_time_serie,
    fill_time_serie,
    keypoints_as_time_series,
)


def get_score(chore: Choregraphy, trainee: Choregraphy, method: str):
    ct_keypoints, ct_visible = keypoints_as_time_series(chore.keypoints)
    tt_keypoints, tt_visible = keypoints_as_time_series(trainee.keypoints)

    cap = cv2.VideoCapture(chore.video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    cap.release()

    nct_keypoints = {
        name: fill_time_serie(np.array(sequence), ct_visible[name], fps)
        for name, sequence in ct_keypoints.items()
    }
    ntt_keypoints = {
        name: fill_time_serie(np.array(sequence), tt_visible[name], fps)
        for name, sequence in tt_keypoints.items()
    }

    score = 0
    if method == "fast_dtw":
        for name in nct_keypoints.keys():
            nct_vel = make_velocity_sequence(nct_keypoints[name])
            ntt_vel = clip_sequence_by_other(
                make_velocity_sequence(ntt_keypoints[name]), nct_vel
            )
            score += fast_dtw(
                nct_vel,
                ntt_vel,
            )
    else:
        raise NotImplementedError

    score = (1 - score / len(nct_keypoints.keys()) / chore.baseline_score) * 100

    return score


def make_velocity_sequence(sequence: np.ndarray):
    velocity_sequence = []
    frame_1 = sequence[0]
    for i in range(1, len(sequence)):
        frame_2 = sequence[i]
        velocity = frame_2 - frame_1
        velocity_sequence.append(velocity)
        frame_1 = frame_2
    return np.array(velocity_sequence)


def clip_sequence_by_other(sequence1: np.ndarray, sequence2: np.ndarray) -> np.ndarray:
    return np.clip(
        sequence1,
        a_min=np.min(sequence2, axis=0),
        a_max=np.max(sequence2, axis=0),
    )


def get_sequences_in_shape(
    sequence1: np.ndarray, sequence2: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    if len(sequence1) < len(sequence2):
        sequence1 = extrapolate_time_serie(sequence1, len(sequence2))
    elif len(sequence2) < len(sequence1):
        sequence2 = extrapolate_time_serie(sequence2, len(sequence1))

    sequence1 = np.where(sequence1 >= 0, sequence1, 0)
    sequence2 = np.where(sequence2 >= 0, sequence2, 0)

    return sequence1, sequence2


def mse(sequence1: np.ndarray, sequence2: np.ndarray) -> float:
    """Computes the Mean Square Error between two sequences
    Args:
        sequence1, sequence2 (np.ndarray) [T, 2]
    Returns:
        mse (float)
    """
    return np.mean(np.square(np.subtract(get_sequences_in_shape(sequence1, sequence2))))


def pearson(sequence1: np.ndarray, sequence2: np.ndarray) -> float:
    """Computes the Pearson's correlation between two sequences
    Args:
        sequence1, sequence2 (np.ndarray) [T, 2]
    Returns:
        pearson (float)
    """
    sequence1, sequence2 = get_sequences_in_shape(sequence1, sequence2)
    mean1, mean2 = np.mean(sequence1, axis=0), np.mean(sequence2, axis=0)

    return np.divide(
        np.sum(np.multiply(sequence1 - mean1, sequence2 - mean2)),
        np.multiply(
            np.sqrt(np.sum(np.square(sequence1 - mean1))),
            np.sqrt(np.sum(np.square(sequence2 - mean2))),
        ),
    )


def cosine(sequence1: np.ndarray, sequence2: np.ndarray) -> float:
    """Computes the cosine between two sequences
    Args:
        sequence1, sequence2 (np.ndarray) [T, 2]
    Returns:
        cosine (float)
    """
    sequence1, sequence2 = get_sequences_in_shape(sequence1, sequence2)

    return np.divide(
        np.sum(sequence1 * sequence2),
        np.multiply(
            np.sqrt(np.sum(np.square(sequence1))), np.sqrt(np.sum(np.square(sequence2)))
        ),
    )


def dtw(sequence1: np.ndarray, sequence2: np.ndarray) -> float:
    """Computes the dtw between two sequences
    Args:
        sequence1, sequence2 (np.ndarray) [T, 2]
    Returns:
        dtw (float)
    """
    sequence1, sequence2 = get_sequences_in_shape(sequence1, sequence2)
    T = len(sequence1)
    C = sequence1.shape[-1]
    dtw_matrix = np.zeros((T + 1, T + 1, C))
    dtw_matrix[0] = np.inf * np.ones_like(dtw_matrix[0])
    dtw_matrix[:, 0] = np.inf * np.ones_like(dtw_matrix[:, 0])
    dtw_matrix[0, 0] = np.zeros((C,))

    for i in range(1, T + 1):
        for j in range(1, T + 1):

            cost = np.abs(sequence1[i - 1] - sequence2[j - 1])
            last_min = np.min(
                [dtw_matrix[i - 1, j], dtw_matrix[i, j - 1], dtw_matrix[i - 1, j - 1]],
                axis=0,
            )
            dtw_matrix[i, j] = cost + last_min

    return np.sum(dtw_matrix[-1, -1])


def fast_dtw(s, t):
    return np.sum([fastdtw(s[..., i], t[..., i])[0] for i in range(s.shape[-1])])


def get_dtw_from_keypoints(nct_keypoints, chal_keypoints) -> float:
    chal_score = 0
    for name in nct_keypoints.keys():
        nct_vel = make_velocity_sequence(nct_keypoints[name])
        ntt_vel = clip_sequence_by_other(
            make_velocity_sequence(chal_keypoints[name]), nct_vel
        )
        chal_score += fast_dtw(
            nct_vel,
            ntt_vel,
        )
    return chal_score / len(nct_keypoints.keys()) / 3


def find_baseline_sequence_dtw(
    chore_keypoints: List[Dict[str, List[float]]],
    chore_video_path: str,
    baseline_keypoints: Optional[Dict[str, np.ndarray]] = None,
) -> Tuple[Dict[str, np.ndarray], float]:

    ct_keypoints, ct_visible = keypoints_as_time_series(chore_keypoints)

    cap = cv2.VideoCapture(chore_video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    cap.release()

    nct_keypoints = {
        name: fill_time_serie(np.array(sequence), ct_visible[name], fps)
        for name, sequence in ct_keypoints.items()
    }

    baseline_keypoints = (
        baseline_keypoints
        if baseline_keypoints is not None
        else {
            name: np.random.uniform(0, 1, size=nct_keypoints[name].shape)
            for name in ct_keypoints.keys()
        }
    )
    baseline_score = get_dtw_from_keypoints(nct_keypoints, baseline_keypoints)

    return baseline_keypoints, baseline_score
