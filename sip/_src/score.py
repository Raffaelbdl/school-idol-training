import numpy as np

from sip._src.chore import Choregraphy
from sip._src.keypoint import keypoints_to_time_series
from sip._src.metadata import JOINT_PAIRS
from sip._src.metric import cosine
from sip._src.sequence import interpolate_time_series, union_of_masks


def cosine_similarity(chore1: Choregraphy, chore2: Choregraphy) -> float:
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
    count = np.sum([len(m) for m in mask.values()])

    cosines = []
    link_masks = []
    for joint_pair in JOINT_PAIRS:
        sequence1 = t_keypoints1[joint_pair[0]] - t_keypoints1[joint_pair[1]]
        sequence2 = new_t_keypoints2[joint_pair[0]] - new_t_keypoints2[joint_pair[1]]
        _mask = union_of_masks(mask[joint_pair[0]], mask[joint_pair[1]])
        cosines.append(cosine(sequence1, sequence2, _mask))
        link_masks.append(_mask)

    normalized = (np.sum(cosines) + np.sum(link_masks)) / (2 * np.sum(link_masks))
    return normalized
