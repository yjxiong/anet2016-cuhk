import os
from easydict import EasyDict

"""
This file contains the setting for running ActivityNet related experiments.
"""

ANET_CFG = EasyDict()
ANET_CFG.ANET_HOME = os.getenv("ANET_HOME",None)
if ANET_CFG.ANET_HOME is None:
    raise ValueError("To use this package, "
                     "set the environmental variable \"ANET_HOME\" to the root director of the codebase")


# Version and other macro settings

ANET_CFG.DB_VERSIONS = {
    '1.2': 'data/activity_net.v1-2.min.json',
    '1.3': 'data/activity_net.v1-3.min.json'
}

# Force the leaf node to be included in the label list
ANET_CFG.FORCE_INCLUDE = {"1.3": [], "1.2": []}

# Acceptable extension of the video files
ANET_CFG.ACC_EXT = {'.mp4', '.webm', '.avi', '.mkv'}

# File name pattern of the video files
ANET_CFG.SRC_ID_LEN = 11 # length of youtube IDs

# Max length of video, -1 for unlimited
ANET_CFG.MAX_DURATION = -1

# MISC
ANET_CFG.CAFFE_ROOT='lib/caffe-action/'
ANET_CFG.DENSE_FLOW_ROOT='lib/dense_flow/'


# Allow using external config files to override the above settings
def LoadExternalYAMLConfig(yaml_file):
    import yaml
    new_cfg = yaml.load(open(yaml_file))
    ANET_CFG.update(new_cfg)
