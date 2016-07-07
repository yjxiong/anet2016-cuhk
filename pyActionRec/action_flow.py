from config import ANET_CFG

import sys

sys.path.append(ANET_CFG.DENSE_FLOW_ROOT+'/build')

from libpydenseflow import TVL1FlowExtractor
import action_caffe
import numpy as np


class FlowExtractor(object):

    def __init__(self, dev_id, bound=20):
        TVL1FlowExtractor.set_device(dev_id)
        self._et = TVL1FlowExtractor(bound)

    def extract_flow(self, frame_list, new_size=None):
        """
        This function extracts the optical flow and interleave x and y channels
        :param frame_list:
        :return:
        """
        frame_size = frame_list[0].shape[:2]
        rst = self._et.extract_flow([x.tostring() for x in frame_list], frame_size[1], frame_size[0])
        n_out = len(rst)
        if new_size is None:
            ret = np.zeros((n_out*2, frame_size[0], frame_size[1]))
            for i in xrange(n_out):
                ret[2*i, :] = np.fromstring(rst[i][0], dtype='uint8').reshape(frame_size)
                ret[2*i+1, :] = np.fromstring(rst[i][1], dtype='uint8').reshape(frame_size)
        else:
            import cv2
            ret = np.zeros((n_out*2, new_size[1], new_size[0]))
            for i in xrange(n_out):
                ret[2*i, :] = cv2.resize(np.fromstring(rst[i][0], dtype='uint8').reshape(frame_size), new_size)
                ret[2*i+1, :] = cv2.resize(np.fromstring(rst[i][1], dtype='uint8').reshape(frame_size), new_size)

        return ret


if __name__ == "__main__":
    import cv2
    im1 = cv2.imread('../data/img_1.jpg')
    im2 = cv2.imread('../data/img_2.jpg')

    f = FlowExtractor(0)
    flow_frames = f.extract_flow([im1, im2])
    from pylab import *

    plt.figure()
    plt.imshow(flow_frames[0])
    plt.figure()
    plt.imshow(flow_frames[1])
    plt.figure()
    plt.imshow(im1)
    plt.show()

    print flow_frames
