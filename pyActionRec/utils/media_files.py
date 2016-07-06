"""
This file provides some utilities to work with media files on the harddisk
"""

import glob
import os


def get_all_media_files(src_folders, accepted_extensions):
    media_files = []
    for f in src_folders:
        all_files = glob.glob(os.path.join(f, "*"))
        media_files.extend([name for name in all_files if os.path.splitext(name)[-1] in accepted_extensions])
    return media_files
