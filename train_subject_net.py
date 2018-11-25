import torch
from subject_clf.subject_net import SubjectNet, train_subject_net
from subject_clf.dataset_imperial import dataset, subjects, short_term_weights_path
from test_short_term import run_evaluation
torch.manual_seed(1234)

if __name__ == '__main__':
    num_joints = 27
    prefix_length = 50
    subject_net = SubjectNet(num_joints, len(subjects), prefix_length, short_term_weights_path)
    if torch.cuda.is_available():
        subject_net.cuda()

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
    train_subject_net(subject_net, batch_size, sequences_train, sequences_valid, dataset, n_epochs)

    # Save weights
    subject_weights_path = 'weights_subject.bin'
    subject_net.save_weights(subject_weights_path)
