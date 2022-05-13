from dataclasses import dataclass, field
from typing import Dict, List

from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmarkList
import numpy as np


@dataclass
class Choregraphy:
    """A choregraphy is described by its title and its video"""

    title: str
    video: np.ndarray
    keypoints: List[Dict[str, List[float]]]
    landmarks: List[NormalizedLandmarkList]

    original_video_path: str = field(default=None)
