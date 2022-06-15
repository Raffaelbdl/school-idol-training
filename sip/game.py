import time
from typing import Dict, List, Tuple

import cv2
from mediapipe.python.solutions import pose as mp_pose
import numpy as np
import pyglet

from sip._src.chore import Choregraphy
from sip._src.keypoint import keypoints_to_time_series, split_keypoint
from sip._src.metadata import LANDMARK_NAMES
from sip._src.score import alt_cosine_similarity
from sip._src.sequence import split_sequence
from sip._src.video import get_duration, test_camera

import pickle


def get_chore_sequence_splits(
    chore: Choregraphy, split_duration: float
) -> Tuple[Dict[str, List[np.ndarray]]]:
    """
    Args:
        split_duration (float): Duration in seconds
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


def capture_keypoints(vid: cv2.VideoCapture):
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:

        ret, frame = vid.read()
        frame = cv2.flip(frame, 0)
        frame = cv2.flip(frame, 1)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame)
        pose_landmarks = results.pose_landmarks

        if pose_landmarks is not None:
            frame_landmarks = {}
            for idx, landmark in enumerate(pose_landmarks.landmark):
                landmark_name = LANDMARK_NAMES[idx]
                if landmark.HasField("visibility") and landmark.visibility >= 0.5:
                    frame_landmarks[landmark_name] = [
                        landmark.x,
                        landmark.y,
                        landmark.z,
                    ]
            return frame_landmarks
        else:
            return None


def fill_buffer(
    dt,
    vid: cv2.VideoCapture,
):
    frame_keypoints = capture_keypoints(vid)
    global BUFFER
    BUFFER.append(frame_keypoints)


def score(
    dt,
    chore_buffers: List[List[Dict[str, List[float]]]],
    label,
):

    global AT, BUFFER, LAST, SCORE
    # pickle.dump(BUFFER, open(f"buffer_{AT}", "wb"))
    # pickle.dump(chore_buffers[AT], open(f"choreat_{AT}", "wb"))

    SCORE, visibility = alt_cosine_similarity(BUFFER, chore_buffers[AT], 0)

    AT += 1
    BUFFER = []
    # print("score ", SCORE)
    if SCORE > 0.80:
        LAST = "good"
    else:
        LAST = "ok"

    label.text = f"{SCORE:.2f} - {visibility:.2f}"


def launch_game(chore: Choregraphy):

    fps = test_camera(True)
    time.sleep(3)
    vid = cv2.VideoCapture(0)

    title = "School Idol Project"
    window = pyglet.window.Window(fullscreen=True, caption=title)

    video_path = chore.video_path
    video_player = pyglet.media.Player()
    video_media = pyglet.media.load(video_path)
    video_player.queue(video_media)
    video_player.play()

    img = pyglet.image.load("./resources/kotori.jpeg")

    global SCORE
    label = pyglet.text.Label(
        "",
        font_name="Times New Roman",
        font_size=36,
        x=0,
        y=50,
        anchor_x="left",
        anchor_y="center",
    )

    global BUFFER, AT, LAST
    BUFFER, AT, LAST = [], 0, "ok"
    n_splits = int(get_duration(video_path) // 1) + 1
    chore_buffers = split_keypoint(chore.keypoints, n_splits)

    pyglet.clock.schedule_interval(fill_buffer, 1 / fps, vid)
    pyglet.clock.schedule_interval(score, 1, chore_buffers, label)

    @window.event
    def on_draw():
        window.clear()

        if video_player.source and video_player.source.video_format:
            texture = video_player.get_texture()
            w = texture.width
            texture.blit(960 - w // 2, 0)
        else:
            pyglet.app.exit()

        global LAST
        if LAST == "good":
            img.blit(0, 850)

        label.draw()

    pyglet.app.run()
