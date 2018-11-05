#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 13:45:06 2018

@author: mollgaard
"""

import ujson
import numpy as np
from common.quaternion import expmap_to_quaternion,qfix
import sys

out_pos = []
out_rot = []
out_subjects = []
out_actions = []

assert len(sys.argv)>2
filename = sys.argv[1]
output_file_path = sys.argv[2]

print('Loading file...')
with open(filename) as file:
    data = ujson.load(file)

for subject in data:
    print('Loading subject: ',subject['info'],'...',sep='')
    for i, half in enumerate(subject['trialData'][1:]):
        
        exp_data = np.array(half['expChannels']).squeeze()
        exp_data = exp_data[:,3:].reshape([-1,27,3])

        exp_data = exp_data[:,:,[2,1,0]]
        
        quat_data = expmap_to_quaternion(exp_data)
        quat_data = qfix(quat_data)
        
        out_pos.append(np.zeros((quat_data.shape[0],3)))
        out_subjects.append(subject['info'])
        out_rot.append(quat_data)
        out_actions.append('walking_'+str(i+1))

print('Saving data...')
np.savez_compressed(output_file_path,
                    trajectories = out_pos,
                    rotations = out_rot,
                    subjects = out_subjects,
                    actions = out_actions)
        
    

    
    
    



