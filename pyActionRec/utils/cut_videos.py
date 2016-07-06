import json
import cv2
import os
import sys
import math
import logging

import numpy as np


def cut_video_normalized(all_frames, out_name, starting, ending, total_duration, size=None):
    """
    Cut the video according to the portion of frames.
    We first read in all frames to memory, then the activity instances are cropped based on the
    portion of time compared with the total length of the video
    :param all_frames: list of video frames
    :param out_name: output activity clip filename
    :param starting: starting time in second
    :param ending: ending time in second
    :param total_duration:  total_duration of the video
    :return:
    """

    # open video streams I/O
    if size is None:
        width = all_frames[0].shape[1]
        height = all_frames[0].shape[0]
        new_size = (width, height)
    else:
        new_size = size

    frame_cnt = len(all_frames)
    anno_fps = int(frame_cnt / total_duration) #seems DIVX does not like some high denomitor FPS

    out_video = cv2.VideoWriter(out_name, cv2.cv.FOURCC(*'DIVX'), anno_fps, new_size, True)
    if not out_video.isOpened():
        print "open instance output video failed", anno_fps, new_size, out_name
        import sys
        sys.stdout.flush()
        raise

    starting_index = int(np.floor(starting / total_duration * frame_cnt))
    ending_index = max(int(np.ceil(ending / total_duration * frame_cnt)), starting_index + 1)

    out_frames = all_frames[starting_index:ending_index]
    for frame in out_frames:
        if size:
            resize_frame = cv2.resize(frame, size, interpolation=cv2.INTER_LINEAR)
        else:
            resize_frame = frame
        out_video.write(resize_frame)

    out_video.release()
    return 0


