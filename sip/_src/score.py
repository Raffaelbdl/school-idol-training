from typing import List, Optional, Tuple

import numpy as np

from sip._src import JOINT_PAIRS, Keypoints
from sip._src.choregraphy import Choregraphy
from sip._src.keypoint import keypoints_to_time_series
from sip._src.sequence import interpolate_time_series, union_of_masks


def cosine(
    sequence1: np.ndarray, sequence2: np.ndarray, mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """Compute the cosine between two sequences

    Args:
        sequences (np.ndarray) [T, D] with D in {2, 3}
        mask (np.ndarray) [T, ]: the union of masks for both sequences

    Returns:
        cosine (np.ndarray) [T, ]
    """
    assert len(sequence1) == len(sequence2)

    norm1 = np.sqrt(np.sum(np.square(sequence1), axis=-1))
    norm2 = np.sqrt(np.sum(np.square(sequence2), axis=-1))
    mask = mask if mask is not None else np.ones_like(norm1)
    assert len(mask) == len(sequence1) and mask.ndim == 1

    return (np.sum(sequence1 * sequence2, axis=-1) / (norm1 * norm2 + 1e-6)) * mask


def score_modifier(x: float, difficulty: int = 0) -> float:
    """Modify the score by a gaussian"""

    def gauss(x, a=1, b=1, c=0.2):
        return a * np.exp(-((x - b) ** 2) / (2 * c**2))

    if difficulty == 0:  # easy
        return (1 / 3) * (
            gauss(x, 1, 1, 0.4) + gauss(x, 1, 1, 0.3) + gauss(x, 1, 1, 0.4)
        )
    elif difficulty == 1:  # medium
        return (1 / 3) * (
            gauss(x, 1, 1, 0.4) + gauss(x, 1, 1, 0.2) + gauss(x, 1, 1, 0.2)
        )
    elif difficulty == 2:  # hard
        return (1 / 3) * (
            gauss(x, 1, 1, 0.2) + gauss(x, 1, 1, 0.2) + gauss(x, 1, 1, 0.1)
        )
    else:
        raise ValueError(f"{difficulty} difficulty is unknown")


def cosine_similarity(
    chore1: Choregraphy,
    chore2: Choregraphy,
    difficulty: int,
) -> Tuple[float, float]:
    """Compute the cosine similarity between chore1 and chore2"""
    keypoints1 = chore1.keypoints
    keypoints2 = chore2.keypoints
    return alt_cosine_similarity(keypoints1, keypoints2, difficulty)


def alt_cosine_similarity(
    k1: List[Keypoints],
    k2: List[Keypoints],
    difficulty: int,
) -> Tuple[float, float]:
    """Compute the cosine similarity between keypoints lists

    * k2 is the list that will be interpolated,
    therefore it should be the user's keypoints list

    Args:
        k1 and k2 (List[Keypoints])
        difficulty (float)

    Returns:
        score (float) and visibility (float)
    """
    t_keypoints1, t_visible1 = keypoints_to_time_series(k1)
    t_keypoints2, t_visible2 = keypoints_to_time_series(k2)

    new_t_keypoints2, new_t_visible2 = {}, {}
    for (name, seq) in t_keypoints2.items():
        new_seq, new_mask = interpolate_time_series(
            seq, len(t_keypoints1[name]), t_visible2[name]
        )
        new_t_keypoints2[name] = new_seq
        new_t_visible2[name] = new_mask

    mask = {
        name: union_of_masks(t_visible1[name], new_t_visible2[name])
        for name in new_t_visible2.keys()
    }

    cosines = []
    link_masks = []
    count_masks = {}
    link_count_masks = {}
    for joint_pair in JOINT_PAIRS:

        sequence1 = t_keypoints1[joint_pair[0]] - t_keypoints1[joint_pair[1]]
        sequence2 = new_t_keypoints2[joint_pair[0]] - new_t_keypoints2[joint_pair[1]]
        _mask = union_of_masks(mask[joint_pair[0]], mask[joint_pair[1]])

        cosines.append(cosine(sequence1, sequence2, _mask))
        link_masks.append(_mask)

        if joint_pair[0] not in count_masks.keys():
            count_masks[joint_pair[0]] = new_t_visible2[joint_pair[0]]
        if joint_pair[1] not in count_masks.keys():
            count_masks[joint_pair[1]] = new_t_visible2[joint_pair[1]]

        if joint_pair[0] not in link_count_masks.keys():
            link_count_masks[joint_pair[0]] = mask[joint_pair[0]]
        if joint_pair[1] not in link_count_masks.keys():
            link_count_masks[joint_pair[1]] = mask[joint_pair[1]]

    normalized = (np.sum(cosines) + np.sum(link_masks)) / (
        2 * np.sum(link_masks) + 1e-3
    )
    normalized = score_modifier(normalized, difficulty=difficulty)

    link_count = np.sum([s for s in link_count_masks.values()])
    count = np.sum([s for s in count_masks.values()])

    return normalized, link_count / count
