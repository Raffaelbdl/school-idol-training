import pickle
from typing import Dict, List, Optional, Tuple

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from sip._src.chore import Choregraphy
from sip._src.sequence import interpolate_time_series


def get_keypoints_from_blender_output(
    blender_output_path: str,
    blender_keypoint_names: List[str],
    blender_name_correspondance: Dict[str, str],
    axes: Optional[str] = "xz",
    stop: Optional[int] = -1,
):
    """Generates keypoints from blender output file

    Args:
        blender_output_path (str)
        blender_keypoint_names (List[str]): the name of the bones
            from which you want the head
        blender_name_correspondance (Dict[str, str]): correspondance
            between blender names and LANDMARK_NAMES

    Returns:
        keypoints (List[Dict[str, List[float]]])
    """
    xyz = {"x": 0, "y": 1, "z": 2}

    with open(blender_output_path, "rb") as file:
        output = pickle.load(file)

    keypoints = []
    n_frames = len(output[next(iter((output.keys())))]["pos"])
    if stop > -1:
        n_frames = min(stop, n_frames)
    for frame in range(n_frames):
        frame_dict = {}
        for name in blender_keypoint_names:
            new_name = blender_name_correspondance[name]
            x = output[name]["pos"][frame, xyz[axes[0]]]
            y = output[name]["pos"][frame, xyz[axes[1]]]
            frame_dict[new_name] = [x, y]
        keypoints.append(frame_dict)

    return keypoints


def animation_from_blender_output(
    blender_output_path: str,
    blender_keypoint_names: List[str],
    blender_name_correspondance: Dict[str, str],
):
    keypoints = get_keypoints_from_blender_output(
        blender_output_path,
        blender_keypoint_names,
        blender_name_correspondance,
    )

    fig, ax = plt.subplots()
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)

    scats = []
    for name in blender_keypoint_names:
        new_name = blender_name_correspondance[name]
        scat = ax.scatter([], [], label=new_name)
        scats.append(scat)
    ax.legend()

    def init():
        frame_keypoints = keypoints[0]
        for scat, name in zip(scats, blender_keypoint_names):
            new_name = blender_name_correspondance[name]
            array = frame_keypoints[new_name]
            scat.set_offsets(array)
        return (scats,)

    def update(frame):
        frame_keypoints = keypoints[frame]
        for scat, name in zip(scats, blender_keypoint_names):
            new_name = blender_name_correspondance[name]
            array = frame_keypoints[new_name]
            scat.set_offsets(array)
        return (scats,)

    ani = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=len(keypoints),
    )
    plt.show()
