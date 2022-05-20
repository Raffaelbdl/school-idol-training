from typing import Optional
import numpy as np


def cosine(
    sequence1: np.ndarray, sequence2: np.ndarray, mask: Optional[np.ndarray]
) -> np.ndarray:
    """Computes the cosine between two sequences

    Args:
        sequences (np.ndarray) [T, D] with D in {2, 3}
        mask (np.ndarray): Union of masks for both sequences
    Returns:
        cosine (np.ndarray) [T, ]
    """
    norm1 = np.sqrt(np.sum(np.square(sequence1), axis=-1))
    norm2 = np.sqrt(np.sum(np.square(sequence2), axis=-1))
    return (np.sum(sequence1 * sequence2, axis=-1) / (norm1 * norm2 + 1e-6)) * mask
