from dataclasses import dataclass, field
import os
import pickle
import shutil
from typing import Dict, List, Optional

import matplotlib.animation as animation
import matplotlib.pyplot as plt
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmarkList

from sip._src.keypoint import get_keypoints_from_video_file, keypoints_to_time_series


@dataclass
class Choregraphy:
    """A choregraphy is described by its title and its video"""

    title: str
    keypoints: List[Dict[str, List[float]]]

    landmarks: List[NormalizedLandmarkList] = field(default=None)
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


def plot_chore(chore: Choregraphy, keys: Optional[List[str]] = None) -> None:
    keypoints = chore.keypoints
    return plot_keypoints(keypoints, keys)


def plot_keypoints(keypoints, keys: Optional[List[str]] = None) -> None:

    t_keypoints, t_visible = keypoints_to_time_series(keypoints)

    if keys is None:
        names = list(t_keypoints.keys())
    else:
        for name in keys:
            if name not in t_keypoints.keys():
                print(f"keys contains invalid name {name}")
                break
        else:
            names = keys

    fig, ax = plt.subplots()
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)

    scats = []
    for name in names:
        scat = ax.scatter([], [], label=name)
        scats.append(scat)
    ax.legend()

    prev_keypoints = {}
    for name in names:
        i, b = 0, 0
        while b == 0 and i < len(t_visible[name]):
            b = t_visible[name][i]
            i += 1
        if b == 1:
            prev_keypoints[name] = [t_keypoints[name][i][:2]]
        else:
            prev_keypoints[name] = [-10, -10]

    def init():
        frame = 0
        for scat, name in zip(scats, names):
            try:
                array = t_keypoints[name][frame][:2]
            except KeyError:
                array = prev_keypoints[name][-1]
            scat.set_offsets(array)
            prev_keypoints[name].append(array)
        return (scats,)

    def update(frame):

        for scat, name in zip(scats, names):
            try:
                array = t_keypoints[name][frame][:2]
            except KeyError:
                array = prev_keypoints[name][-1]
            scat.set_offsets(array)
            prev_keypoints[name].append(array)
        return (scats,)

    ani = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=len(keypoints),
    )
    plt.show()
