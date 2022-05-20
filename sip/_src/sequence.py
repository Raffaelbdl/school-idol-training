from typing import Optional, Tuple

import numpy as np
from scipy.interpolate import interp1d


def interpolate_time_series(
    sequence: np.ndarray,
    new_len: int,
    mask: Optional[np.ndarray] = None,
    mask_threshold: float = 0.3,
) -> Tuple[np.ndarray]:
    """Uses interpolation to augment the length of a time series"""

    cur_T = np.linspace(0, 1, len(sequence))
    new_T = np.linspace(0, 1, new_len)

    new_sequence = np.empty(shape=(new_len, *sequence.shape[1:]))
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
    """Computes union of masks of same length"""
    assert mask1.shape == mask2.shape
    return mask1 * mask2
