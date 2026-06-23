import os
import numpy as np
import imageio

def _make_dir(filename):
    folder = os.path.dirname(filename)
    if not os.path.exists(folder):
        os.makedirs(folder)

def save_video(filename, video_frames, fps=60):
    assert fps == int(fps), fps
    _make_dir(filename)

    video_frames = np.asarray(video_frames)
    if video_frames.dtype != np.uint8:
        video_frames = video_frames.astype(np.uint8)

    imageio.mimwrite(filename, video_frames, fps=int(fps))

def save_videos(filename, *video_frames, **kwargs):
    ## video_frame : [ N x H x W x C ]
    video_frames = np.concatenate(video_frames, axis=2)
    save_video(filename, video_frames, **kwargs)
