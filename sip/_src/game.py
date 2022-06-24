import os
import pickle
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pyglet

from sip._src import Keypoints, CAMERA_LANDMARK_NAMES
from sip._src.choregraphy import Choregraphy
from sip._src.keypoint import (
    keypoints_to_time_series,
    split_keypoint,
    get_keypoints_from_stream,
)
from sip._src.score import alt_cosine_similarity
from sip._src.sequence import split_sequence
from sip._src.video import get_duration, test_camera


def get_chore_sequence_splits(
    chore: Choregraphy, split_duration: float
) -> Tuple[Dict[str, List[np.ndarray]]]:
    """Get segments of keypoints time series from a choregraphy
    Args:
        chore (Choregraphy)
        split_duration (float): duration in seconds of a segment

    Returns:
        split_time_keypoints (List[Dict[str, np.ndarray]])
            where keys are landmarks names
            and values are sequences of list for coordinates
        split_time_visible (List[Dict[str, np.ndarray]])
            where keys are landmarks names
            and values are booleans to tell if joint is visible
    """
    keypoints = chore.keypoints
    t_keypoints, t_visible = keypoints_to_time_series(keypoints)

    length = get_duration(chore.video_path)
    n_splits = int(length / split_duration * 1000) + 1

    split_t_keypoints = {
        key: split_sequence(t_keypoints[key], n_splits) for key in t_keypoints
    }
    split_t_visible = {
        key: split_sequence(t_visible[key], n_splits) for key in t_visible
    }

    return split_t_keypoints, split_t_visible


def _fill_buffer(dt, vid: cv2.VideoCapture) -> None:
    """Capture keypoints and add to global buffer

    Args:
        dt: for pyglet use
        vid: the user's camera cv2 stream
    """
    global BUFFER

    frame_keypoints = get_keypoints_from_stream(vid, CAMERA_LANDMARK_NAMES)[0]
    BUFFER.append(frame_keypoints)


def _score(
    dt,
    chore_buffers: List[List[Keypoints]],
    score_label: pyglet.text.Label,
    difficulty: int,
    save_segments: bool = False,
    save_path: str = "./",
) -> None:
    """Score a segment

    Args:
        dt: for pyglet use
        chore_buffers (List[List[Keypoints]]): from the original choregraphy
        score_label (pyglet.text.Label): the score pyglet text Label
        difficulty (int): the game's difficulty
        save_segments (bool): if True, every segment will be saved on disk
        save_path (str): the path where segments are saved ("./" by default)
    """
    global BUFFER, CUR_SEGMENT, PREV_SCORE, TOTAL_SCORE, TOTAL_VISIBILITY

    if save_segments:
        pickle.dump(
            BUFFER, open(os.path.join(save_path, f"buffer_{CUR_SEGMENT}"), "wb")
        )
        pickle.dump(
            chore_buffers[CUR_SEGMENT],
            open(os.path.join(save_path, f"choreat_{CUR_SEGMENT}"), "wb"),
        )

    score, visibility = alt_cosine_similarity(
        BUFFER, chore_buffers[CUR_SEGMENT], difficulty
    )

    BUFFER = []
    CUR_SEGMENT += 1
    PREV_SCORE = 1 if score > 0.80 else 0
    TOTAL_SCORE += score
    TOTAL_VISIBILITY += visibility

    score_label.text = f"{score:.2f} - {visibility:.2f}"


def launch_game(chore: Choregraphy, difficulty: int):
    """Dance along and score in real time"""
    fps = test_camera(True)
    time.sleep(3)

    # defining global variables
    global BUFFER, CUR_SEGMENT, PREV_SCORE, TOTAL_SCORE, TOTAL_VISIBILITY

    BUFFER = []  # buffer to store current segment
    CUR_SEGMENT = 0  # current segment
    PREV_SCORE = 0  # how well user performed last segment
    TOTAL_SCORE = 0.0
    TOTAL_VISIBILITY = 0.0

    # making window
    title = "School Idol Project"
    window = pyglet.window.Window(fullscreen=True, caption=title)

    # loading user's camera
    vid = cv2.VideoCapture(0)

    # loading choregraphy's video
    video_path = chore.video_path
    video_player = pyglet.media.Player()
    video_media = pyglet.media.load(video_path)
    video_player.queue(video_media)
    video_player.play()

    # loading good segment gif
    good_animation = pyglet.image.load_animation("./resources/test.gif")
    sprite = pyglet.sprite.Sprite(img=good_animation)

    # loading score text canvas
    label = pyglet.text.Label(
        "",
        font_name="Times New Roman",
        font_size=36,
        x=0,
        y=50,
        anchor_x="left",
        anchor_y="center",
    )

    # splitting the chore in segments
    n_splits = int(get_duration(video_path) // 1) + 1
    chore_buffers = split_keypoint(chore.keypoints, n_splits)

    # defining functions to fill buffer and score segment
    pyglet.clock.schedule_interval(_fill_buffer, 1 / fps, vid)
    pyglet.clock.schedule_interval(
        _score, 1, chore_buffers, label, difficulty, save_segments=False
    )

    # defining pyglet on_draw function
    @window.event
    def on_draw():
        window.clear()

        if video_player.source and video_player.source.video_format:
            texture = video_player.get_texture()
            w = texture.width
            texture.blit(960 - w // 2, 0)
        else:
            pyglet.app.exit()

        if PREV_SCORE:
            sprite.draw()

        label.draw()

    pyglet.app.run()
    window.close()
    vid.release()

    return TOTAL_SCORE / n_splits, TOTAL_VISIBILITY / n_splits
