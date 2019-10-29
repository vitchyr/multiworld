import os
import numpy as np
import cv2
import pdb

def save_image(filepath, img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filepath, img)

## from softlearning/misc/utils.py

def make_dir(*folder):
    folder = os.path.join(*folder)
    if not os.path.exists(folder):
        os.makedirs(folder)


def save_video(filename, video_frames):
    video_frames = np.array(video_frames)
    folder = os.path.dirname(filename)
    make_dir(folder)

    video_frames = np.flip(video_frames, axis=-1)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    fps = 30.0
    (height, width, _) = video_frames[0].shape
    writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    for video_frame in video_frames:
        writer.write(video_frame)
    writer.release()