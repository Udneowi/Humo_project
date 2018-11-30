# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import sys
from long_term.pose_network_long_term_subject import PoseNetworkLongTermSubject
from long_term.dataset_h36m import dataset, long_term_weights_path,subjects_train
from long_term.locomotion_utils import build_extra_features
torch.manual_seed(1234)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Must supply subject name')
        sys.exit(-1)

    subject = sys.argv[1]
    subject_weights_path = 'weights_long_term_h36m_' + subject + '.bin'
    prefix_length = 30
    target_length = 60

    model = PoseNetworkLongTermSubject(prefix_length, dataset.skeleton(), long_term_weights_path)
    if torch.cuda.is_available():
        print('GPU:', torch.cuda.get_device_name(0))
        model.cuda()
        dataset.cuda()

    sequences_train = []
    sequences_valid = []
    n_discarded = 0

    for action in [a for a in dataset[subject].keys() if a.startswith('walking') and a.split('_')[1] == '1']:
        if dataset[subject][action]['rotations'].shape[0] < prefix_length + target_length:
            n_discarded += 1
            continue

        sequences_train.append((subject, action))

    print('%d sequences were discarded for being too short.' % n_discarded)
    print('Training on %d sequences, validating on %d sequences.' % (len(sequences_train), len(sequences_valid)))
    dataset.compute_positions()
    build_extra_features(dataset)
    model.train(dataset, target_length, sequences_train, sequences_valid)
    model.save_weights(subject_weights_path)
