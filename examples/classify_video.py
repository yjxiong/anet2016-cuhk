"""
This scripts demos how to do single video classification using the framework
Before using this scripts, please download the model files using

bash models/get_reference_models.sh

Usage:

python classify_video.py <video name>
"""

import os
anet_home = os.environ['ANET_HOME']
import sys
sys.path.append(anet_home)

from pyActionRec.action_classifier import ActionClassifier
from pyActionRec.anet_db import ANetDB
import argparse
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("video_name", type=str)
parser.add_argument("--use_flow", action="store_true", default=False)
parser.add_argument("--gpu", type=int, default=0)

args = parser.parse_args()

VIDEO_NAME = args.video_name
USE_FLOW = args.use_flow
GPU=args.gpu

models=[]

models = [('models/resnet200_anet_2016_deploy.prototxt',
           'models/resnet200_anet_2016.caffemodel',
           1.0, 0, True, 224)]


if USE_FLOW:
    models.append(('models/bn_inception_anet_2016_temporal_deploy.prototxt',
                   'models/bn_inception_anet_2016_temporal.caffemodel.v5',
                   0.2, 1, False, 224))

cls = ActionClassifier(models, dev_id=GPU)
rst = cls.classify(VIDEO_NAME)

scores = rst[0]


db = ANetDB.get_db("1.3")
lb_list = db.get_ordered_label_list()


idx = np.argsort(scores)[::-1]

print '----------------Classification Results----------------------'
for i in xrange(10):
    k = idx[i]
    print lb_list[k], scores[k]


