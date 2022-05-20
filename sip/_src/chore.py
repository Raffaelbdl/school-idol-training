from dataclasses import dataclass, field
import os
import pickle
import shutil
from typing import Dict, List, Optional

from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmarkList

from sip._src.keypoint import get_keypoints_from_video_file


@dataclass
class Choregraphy:
    """A choregraphy is described by its title and its video"""

    title: str
    keypoints: List[Dict[str, List[float]]]
    landmarks: List[NormalizedLandmarkList]
    video_path: str = field(default=None)

    score: float = field(default=None)  # for trainee


def load_chore(chore_path: str) -> Choregraphy:
    chore_path = chore_path.rstrip("/")
    title = chore_path.split("/")[-1]
    with open(os.path.join(chore_path, "keypoints"), "rb") as f:
        keypoints = pickle.load(f)
    with open(os.path.join(chore_path, "landmarks"), "rb") as f:
        landmarks = pickle.load(f)
    with open(os.path.join(chore_path, "score"), "rb") as f:
        score = pickle.load(f)

    return Choregraphy(
        title=title,
        keypoints=keypoints,
        landmarks=landmarks,
        video_path=os.path.join(chore_path, "video.mp4"),
        score=score,
    )


def make_chore_from_file(
    title: str, filepath: str, load_message: Optional[str] = None
) -> Choregraphy:
    """Creates a choregraphy from a video"""
    assert os.path.isfile(filepath)
    keypoints, landmarks = get_keypoints_from_video_file(
        filepath=filepath, load_message=load_message
    )
    return Choregraphy(
        title=title,
        keypoints=keypoints,
        landmarks=landmarks,
        video_path=filepath,
    )


def save_chore(chore: Choregraphy, dirpath: str) -> None:
    chore_path = os.path.join(dirpath, chore.title)
    if not os.path.isdir(chore_path):
        os.makedirs(chore_path)

    shutil.copy(chore.video_path, os.path.join(chore_path, "video.mp4"))

    with open(os.path.join(chore_path, "keypoints"), "wb") as f:
        pickle.dump(chore.keypoints, f)
    with open(os.path.join(chore_path, "landmarks"), "wb") as f:
        pickle.dump(chore.landmarks, f)
    with open(os.path.join(chore_path, "score"), "wb") as f:
        pickle.dump(chore.score, f)
