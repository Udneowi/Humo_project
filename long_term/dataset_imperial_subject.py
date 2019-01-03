# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import re
import numpy as np
import pandas as pd
import os.path
import pickle
from glob import glob
from common.quaternion import expmap_to_quaternion, qfix
from common.skeleton import Skeleton
from common.mocap_dataset import MocapDataset
import numpy as np
import ujson

# Set to True for validation.
# There is no test set here, since we do not evaluate numerically
# for long-term generation of locomotion.
leave_out = []
subject = 'Subject18'
# Note: the "joints_left" and "joint_right" indices refer to the optimized skeleton
# after calling "remove_joints".
with open('skeleton_subject.json') as data_temp:
    skeletons = ujson.load(data_temp)

skeleton_imperial = Skeleton(offsets=skeletons[subject],
    parents=[-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 9, 0, 11, 12, 13, 14, 15, 13, 17, 18, 19, 20, 13, 22, 23, 24, 25],
    joints_left=[1, 2, 3, 4, 5, 17, 18, 19, 20, 21],
    joints_right=[6, 7, 8, 9, 10, 22, 23, 24, 25, 26])

dataset_path = 'datasets/imperial_full.npz'
long_term_weights_path = 'weights_long_term.bin'
dataset = MocapDataset(dataset_path, skeleton_imperial, fps=120,subject_leave_out = leave_out)

# Remove useless joints, from both the skeleton and the dataset
#skeleton_cmu.remove_joints([13, 21, 23, 28, 30], dataset)

#dataset.mirror()
dataset.compute_euler_angles('zyx')
dataset.downsample(8)
