from typing import List, Optional, Tuple

import numpy as np
from scipy.interpolate import interp1d


def interpolate_time_series(
    sequence: np.ndarray,
    new_length: int,
    mask: Optional[np.ndarray] = None,
    mask_threshold: float = 0.3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Use interpolation to change the length of a time series

    Args:
        sequence (np.ndarray) [T, D]
        new_length (int)
        mask (np.ndarray) [T, ]: if given, values under the mask threshold
            will be set to zero (used for not interpolating some values)
        mask_threshold (float)

    Returns:
        new_sequence (np.ndarray) and new_mask (np.ndarray)
    """
    cur_T = np.linspace(0, 1, len(sequence))
    new_T = np.linspace(0, 1, new_length)

    new_sequence = np.empty(shape=(new_length,) + sequence.shape[1:])
    for i in range(new_sequence.shape[-1]):

        interpol_fn = interp1d(cur_T, sequence[..., i])
        new_vals = interpol_fn(new_T)

        if mask is not None:
            interpol_mask = interp1d(cur_T, mask)
            new_mask = np.where(interpol_mask(new_T) < mask_threshold, 0.0, 1.0)
            new_sequence[..., i] = new_vals * new_mask
        else:
            new_sequence[..., i] = new_vals

    return new_sequence, new_mask


def union_of_masks(mask1: np.ndarray, mask2: np.ndarray):
    """Compute union of masks of same length"""
    assert mask1.shape == mask2.shape
    return mask1 * mask2


def split_sequence(sequence: np.ndarray, n_splits: int) -> List[np.ndarray]:
    """Split a sequence"""
    split_sequences_list = []
    split_length = len(sequence) // n_splits

    for i in range(n_splits):
        split_sequences_list.append(sequence[i * split_length : (i + 1) * split_length])

    return split_sequences_list
