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
from common.mocap_dataset import MocapDataset
from common.skeleton import Skeleton

# Set to True for validation, set to False for testing
perform_validation = False

if perform_validation:
    subjects_train = ['Subject1', 'Subject2', 'Subject3', 'Subject4', 'Subject5',
                      'Subject6', 'Subject7', 'Subject8', 'Subject9', 'Subject10',
                      'Subject11', 'Subject12', 'Subject13', 'Subject14', 'Subject15',
                      'Subject16', 'Subject17', 'Subject18', 'Subject19', 'Subject20',
                      'Subject21', 'Subject22', 'Subject23', 'Subject24']
    subjects_valid = ['Subject25']
    subjects_test = ['Subject26']
else:
    subjects_train = ['Subject1', 'Subject2', 'Subject3', 'Subject4', 'Subject5',
                      'Subject6', 'Subject7', 'Subject8', 'Subject9', 'Subject10',
                      'Subject11', 'Subject12', 'Subject13', 'Subject14', 'Subject15',
                      'Subject16', 'Subject17', 'Subject18', 'Subject19', 'Subject20',
                      'Subject21', 'Subject22', 'Subject23', 'Subject24','Subject25',
                      'Subject26']
    subjects_valid = []
    subjects_test = []

dataset_path = 'datasets/data_imperial_2.npz'
short_term_weights_path = 'weights_short_term.bin'

skeleton_imperial = Skeleton(offsets=[
        [0, 0, 0],
        [0.091287263088, 0.547940149596, -0.12057665264400001],
        [0, 0.30301749648, -1.7184998080799998],
        [0, 0.28487456662, -1.61524105617],
        [0.437734481618, 3.412177735459999e-12, -0.125472266928],
        [0.224819, 1.7473719546499998e-12, -9.905952296100001e-12],
        [0.091287246328, -0.528250154848, -0.120576355119999],
        [0, -0.30589658032, -1.7348279247199998],
        [0, -0.28551970826, -1.61915893684],
        [0.46028240332799997, -3.583826328e-12, -0.145672347456],
        [0.237732, -1.8561377590799993e-12, -1.047886996199999e-11],
        [-0.0085707703935, -0.0033291245673, 0.444873776633999],
        [0.0060529383405000005, 0.0014894078199000002, 0.44544134246999995],
        [0.009564423742999002, 0.00298004521605, 0.444805990574],
        [-0.036025562196, 0.00261801257508, 0.35821961838000005],
        [-0.05033537756, 0.0022401767544, 0.353248623519999],
        [-0.013775639145499001, 0.00051162794125, 0.363433503294999],
        [-8.7418796185e-19, 0.756491829177, -0.147047146659],
        [2.1336019996000002e-26, 0.6287597852000001, -1.1343121704],
        [2.33509479224e-26, 0.36122902213999997, -0.65167411428],
        [2.63159004766e-26, 0.18061451106999998, -0.32583705714],
        [-0.055182976822999004, 0.17723818283599999, -0.319275285134],
        [-8.741936553799999e-19, -0.733233159039, -0.14252611821299999],
        [-1.8732430327000002e-26, -0.6255842797000001, -1.1285834094],
        [-2.2757703002599992e-26, -0.35511459842, -0.64064340684],
        [-2.69211306906e-26, -0.17755729921, -0.32032170342],
        [-0.055012222539, -0.16729379953200002, -0.301461351522]],
    parents=[-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 9, 0, 11, 12, 13, 14, 15, 13, 17, 18, 19, 20, 13, 22, 23, 24, 25],
    joints_left=[1, 2, 3, 4, 5, 17, 18, 19, 20, 21],
    joints_right=[6, 7, 8, 9, 10, 22, 23, 24, 25, 26])

dataset = MocapDataset(dataset_path, skeleton_imperial, fps=200)
dataset.downsample(8)
