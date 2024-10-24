import os
import glob
import shutil

import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))

from projects.configs.config import CONF

def main():
    # get poses.txt
    poses_paths = os.path.join(CONF.PATH.DATA_DATASETS_POSES, '*.txt')
    poses_paths = glob.glob(poses_paths)

    # Copy poses to correct dir
    for pose_file in poses_paths:
        file_name = os.path.basename(pose_file)
        prefix = file_name.split('.')[0]

        target_path = os.path.join(CONF.PATH.DATA_DATASETS_SEQUENCES, prefix, 'poses.txt')
        
        shutil.move(pose_file, target_path)


if __name__ == "__main__":
    main()
    