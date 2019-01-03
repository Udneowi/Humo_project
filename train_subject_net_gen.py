import sys
import torch
from subject_clf.subject_net_gen import SubjectNet, train_subject_net
from subject_clf.dataset_imperial import dataset, subjects, short_term_weights_path
from test_short_term import run_evaluation
from long_term.pose_network_long_term import  PoseNetworkLongTerm
from long_term.pace_network import PaceNetwork
from long_term.locomotion_utils import build_extra_features
torch.manual_seed(1234)

if __name__ == '__main__':
    num_joints = 27
    prefix_length = 50
    subject_net = SubjectNet(num_joints, len(subjects), prefix_length, short_term_weights_path)
    pace_net = PaceNetwork()
    pace_net.load_weights('weights_pace_network.bin')
    gen_net = PoseNetworkLongTerm(50, dataset.skeleton())
    gen_net.load_weights('weights_long_term.bin') # Load pretrained model

    if len(sys.argv) > 1 and sys.argv[1] == 'includereal':
        include_real = True
    else:
        include_real = False

    dataset.compute_euler_angles('yzx')
    dataset.compute_positions()
    build_extra_features(dataset)

    if torch.cuda.is_available():
        subject_net.cuda()
        pace_net.cuda()
        gen_net.cuda()

    sequences_train = []
    sequences_valid = []
    for sid, subject in enumerate(subjects):
        for action in dataset[subject].keys():
            if action.startswith('walking_1'):
                sequences_train.append((sid, subject, action))
            else:
                sequences_valid.append((sid, subject, action))

    print('Training on %d sequences, validating on %d sequences.' % (len(sequences_train), len(sequences_valid)))

    batch_size = 60
    n_epochs = 500
    train_subject_net(subject_net, pace_net, gen_net, batch_size, sequences_train, sequences_valid, dataset, include_real, n_epochs, benchmark_every=1)

    # Save weights
    subject_weights_path = 'weights_subject.bin'
    subject_net.save_weights(subject_weights_path)
