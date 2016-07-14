from __future__ import unicode_literals
from action_caffe import CaffeNet
from action_flow import FlowExtractor
from video_proc import VideoProc
from anet_db import Video
from utils.video_funcs import sliding_window_aggregation_func, default_fusion_func
import numpy as np
import time
import youtube_dl
import os


def _dummy_vid_info(url=''):
    info_dict = {
        'annotations': list(),
        'url': url,
        'duration': 10,
        'subset': 'testing'
    }
    return Video('0', info_dict)


class ActionClassifier(object):
    """
    This class provides and end-to-end interface to classifying videos into activity classes
    """

    def __init__(self, models=list(), total_norm_weights=None, score_name='fc-action', dev_id=0):
        """
        Contruct an action classifier
        Args:
            models: list of tuples in the form of
                    (model_proto, model_params, model_fusion_weight, input_type, conv_support, input_size).
                    input_type is: 0-RGB, 1-Optical flow.
                    conv_support indicates whether the network supports convolution testing, which is faster. If this is
                    not supported, we will use oversampling instead
            total_norm_weights: sum of all model_fusion_weights when normalization is wanted, otherwise use None
        """

        self.__net_vec = [CaffeNet(x[0], x[1], dev_id,
                                   input_size=(340, 256) if x[4] else None
                                   ) for x in models]
        self.__net_weights = [float(x[2]) for x in models]

        if total_norm_weights is not None:
            s = sum(self.__net_weights)
            self.__net_weights = [x/s for x in self.__net_weights]

        self.__input_type = [x[3] for x in models]
        self.__conv_support = [x[4] for x in models]

        self.__num_net = len(models)

        # the input size of the network
        self.__input_size = [x[5] for x in models]

        # whether we should prepare flow stack
        self.__need_flow = max(self.__input_type) > 0

        # the name in the proto for action classes
        self.__score_name = score_name

        # the video downloader
        self.__video_dl = youtube_dl.YoutubeDL(
            {
                'outtmpl': '%(id)s.%(ext)s'
            }
        )

        if self.__need_flow:
            self.__flow_extractor = FlowExtractor(dev_id)

    def classify(self, video, model_mask=None):
        """

        Args:
            video:

        Returns:
            scores:
            frm_scores:
        """
        import urlparse

        if os.path.isfile(video):
            return self._classify_from_file(video, model_mask)
        elif urlparse.urlparse(video).scheme != "":
            return self._classify_from_url(video, model_mask)

        raise ValueError("Unknown input data type")

    def _classify_from_file(self, filename, model_mask):
        """
        Input a file on harddisk
        Args:
            filename:

        Returns:
            cls: classification scores
            frm_scores: frame-wise classification scores
        """
        vid_info = _dummy_vid_info()
        vid_info.path = filename
        video_proc = VideoProc(vid_info)
        video_proc.open_video(True)

        # here we use interval of 30, roughly 1FPS
        frm_it = video_proc.frame_iter(timely=False, ignore_err=True, interval=30,
                                       length=6 if self.__need_flow else 1,
                                       new_size=(340, 256))

        all_scores = []
        all_start = time.clock()

        cnt = 0

        # process model mask
        mask = [True] * self.__num_net
        n_model = self.__num_net
        if model_mask is not None:
            for i in xrange(len(model_mask)):
                mask[i] = model_mask[i]
                if not mask[i]:
                    n_model -= 1


        for frm_stack in frm_it:

            start = time.clock()
            cnt += 1
            frm_scores = []

            flow_stack = None
            for net, run, in_type, conv_support, net_input_size in \
                    zip(self.__net_vec, mask, self.__input_type, self.__conv_support, self.__input_size):
                if not run:
                    continue

                frame_size = (340 * net_input_size / 224, 256 * net_input_size / 224)

                if in_type == 0:
                    # RGB input

                    frm_scores.append(net.predict_single_frame(frm_stack[:1], self.__score_name,
                                                               over_sample=not conv_support,
                                                               frame_size=None if net_input_size == 224 else frame_size
                                                               ))
                elif in_type == 1:
                    # Flow input
                    if flow_stack is None:
                        # Extract flow if necessary
                        flow_stack = self.__flow_extractor.extract_flow(frm_stack, frame_size)

                    frm_scores.append(net.predict_single_flow_stack(flow_stack, self.__score_name,
                                                                    over_sample=not conv_support))

            all_scores.append(frm_scores)
            end = time.clock()
            elapsed = end - start
            print "frame sample {}: {} second".format(cnt, elapsed)

        # aggregate frame-wise scores
        agg_scores = []
        for i in xrange(n_model):
            model_scores = sliding_window_aggregation_func(np.array([x[i] for x in all_scores]), norm=False)
            agg_scores.append(model_scores)

        final_scores = default_fusion_func(np.zeros_like(agg_scores[0]), agg_scores, [w for w, m in zip(self.__net_weights, mask) if m])

        all_end = time.clock()
        total_time = all_end - all_start
        print "total time: {} second".format(total_time)

        return final_scores, all_scores, total_time

    def _classify_from_url(self, url, model_mask):
        """
        This function classify an video based on input video url
        It will first use Youtube-dl to download the video. Then will do classification on the downloaded file
        Returns:
            cls: classification scores
            frm_scores: frame-wise classification scores
        """

        file_info = self.__video_dl.extract_info(url) # it also downloads the video file
        filename = file_info['id']+'.'+file_info['ext']

        scores, frm_scores, total_time = self._classify_from_file(filename, model_mask)
        import os
        os.remove(filename)
        return scores, frm_scores, total_time
