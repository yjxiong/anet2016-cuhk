"""
This files provids facilities for processing videos
"""

import cv2


class VideoProc(object):
    """
    The class works with video files and their annotations
    """

    def __init__(self, vid_info, open_on_init=False):
        self._vid_info = vid_info
        self._vid_path = vid_info.path
        self._instances = vid_info.instances

        # video info
        self._vid_cap = None
        self._fps = -1
        self._frame_count = -1
        self._real_fps = -1
        self._frame_width = -1
        self._frame_height = -1

        # length limit
        from config import ANET_CFG
        self._max_frames = 30 * ANET_CFG.MAX_DURATION

        if open_on_init:
            self.open_video()

    def open_video(self, preload=True):
        vcap = cv2.VideoCapture(self._vid_path)

        if not vcap.isOpened():
            raise IOError("Cannot open video file {}, associated to video id {}".format(
                self._vid_path, self._vid_info.id
            ))
        self._frame_width = vcap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
        self._frame_height = vcap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)

        if not preload:
            self._fps = vcap.get(cv2.cv.CV_CAP_PROP_FPS)
            self._frame_count = vcap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
            self._real_fps = self._frame_count / float(self._vid_info.duration)
            self._vid_cap = vcap
            self._loaded = False
        else:
            cnt = 0
            self._frames = []
            cnt = 0
            while True:
                suc, frame = vcap.read()
                cnt += 1

                if 0 < self._max_frames <= cnt:
                    break

                if suc:
                    self._frames.append(frame)
                else:
                    break
            self._frame_count = len(self._frames)
            self._real_fps = self._frame_count / float(self._vid_info.duration)
            self._loaded = True

    def frame_iter(self, starting_frame=0, interval=1, length=1, timely=False, new_size=None, ignore_err=False):
        """
        This is a generator that will return a set of frames according to step and length
        :param starting_frame: the frame index from which the iteration starts
        :param interval: interval of frame sampling
        :param length: how many frame to extract at once
        :param timely: if set to True, the interval will be using the unit of second, instead of frame
        :return: generator of frame stacks
        """
        if not self._loaded:
            self._vid_cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, starting_frame)

            if int(self._vid_cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)) != starting_frame:
                raise IOError("Fail to locate to frame {} of video {} associated with ")

        if timely:
            # calculate the frame interval for the time interval
            frame_interval = int(self._real_fps * interval)
        else:
            frame_interval = interval

        # start the iters
        cur_frame = starting_frame
        while (cur_frame + length) <= self._frame_count:
            frames = []
            for i in xrange(length):
                if self._loaded:
                    frm = self._frames[cur_frame+i]
                    if new_size is not None:
                        frm = cv2.resize(frm, new_size)
                    frames.append(frm.copy())
                else:
                    success, frm = self._vid_cap.read()
                    if not success:
                        if not ignore_err:
                            raise IOError("Read frame failed")
                        else:
                            raise StopIteration
                    if new_size is not None:
                        frm = cv2.resize(frm, new_size)
                    frames.append(frm.copy())
            cur_frame += length
            yield frames

            if not self._loaded:
                skip = frame_interval - length
                if 0 < skip < 100:
                    for i in xrange(skip):
                        self._vid_cap.read()
                    cur_frame += skip
                elif skip < 0 or skip >= 5:
                    self._vid_cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, cur_frame + skip)
                    cur_frame += skip
            else:
                cur_frame += frame_interval - length
