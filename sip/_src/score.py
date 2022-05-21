from ntpath import join
from typing import Tuple

import numpy as np

from sip._src.chore import Choregraphy
from sip._src.keypoint import keypoints_to_time_series
from sip._src.metadata import JOINT_PAIRS
from sip._src.metric import cosine
from sip._src.sequence import interpolate_time_series, union_of_masks


# Could allow for variable difficulty here !
def score_modifier(x: float, difficulty: int = 0):
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


def cosine_similarity(chore1: Choregraphy, chore2: Choregraphy, difficulty: int) -> Tuple[float, float]:
    """Computes the cosine similarity between chore1 and chore2"""
    keypoints1 = chore1.keypoints
    keypoints2 = chore2.keypoints
    t_keypoints1, t_visible1 = keypoints_to_time_series(keypoints1)
    t_keypoints2, t_visible2 = keypoints_to_time_series(keypoints2)

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
        if joint_pair[0] == "neck":
            sequence1 = (t_keypoints1["right_shoulder"] + t_keypoints1["left_shoulder"]) / 2
            sequence1 -= (t_keypoints1["right_hip"] + t_keypoints1["left_hip"]) / 2
            sequence2 = (new_t_keypoints2["right_shoulder"] + new_t_keypoints2["left_shoulder"]) / 2
            sequence2 -= (new_t_keypoints2["right_hip"] + new_t_keypoints2["left_hip"]) / 2
            _mask = union_of_masks(
                union_of_masks(mask["right_shoulder"], mask["left_shoulder"]),
                union_of_masks(mask["right_hip"], mask["left_hip"])
            )
            chore_mask1 = union_of_masks(t_visible1["right_shoulder"], t_visible1["left_shoulder"])
            chore_mask2 = union_of_masks(t_visible1["right_hip"], t_visible1["left_hip"])
            mask1 = union_of_masks(mask["right_shoulder"], mask["left_shoulder"])
            mask2 = union_of_masks(mask["right_hip"], mask["left_hip"])
        else:
            sequence1 = t_keypoints1[joint_pair[0]] - t_keypoints1[joint_pair[1]]
            sequence2 = new_t_keypoints2[joint_pair[0]] - new_t_keypoints2[joint_pair[1]]
            _mask = union_of_masks(mask[joint_pair[0]], mask[joint_pair[1]])
            chore_mask1 = t_visible1[joint_pair[0]]
            chore_mask2 = t_visible2[joint_pair[1]]
            mask1 = mask[joint_pair[0]]
            mask2 = mask[joint_pair[1]]

        cosines.append(cosine(sequence1, sequence2, _mask))
        link_masks.append(_mask)


        if joint_pair[0] not in count_masks.keys():
            count_masks[joint_pair[0]] = chore_mask1
        if joint_pair[1] not in count_masks.keys():
            count_masks[joint_pair[1]] = chore_mask2
        if joint_pair[0] not in link_count_masks.keys():
            link_count_masks[joint_pair[0]] = mask1
        if joint_pair[1] not in link_count_masks.keys():
            link_count_masks[joint_pair[1]] = mask2

    normalized = (np.sum(cosines) + np.sum(link_masks)) / (
        2 * np.sum(link_masks) + 1e-3
    )
    normalized = score_modifier(normalized, difficulty=difficulty)

    link_count = np.sum([s for s in link_count_masks.values()])
    count = np.sum([np.sum(s) for s in count_masks.values()])
    return normalized, link_count / count
