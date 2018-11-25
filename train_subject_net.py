import torch
from subject_clf.subject_net import SubjectNet, train_subject_net
from subject_clf.dataset_h36m import dataset, short_term_weights_path
torch.manual_seed(1234)

if __name__ == '__main__':
    num_joints = dataset.skeleton().num_joints()
    prefix_length = 50
    subject_net = SubjectNet(num_joints, len(dataset.subjects()), prefix_length, short_term_weights_path)
    if torch.cuda.is_available():
        subject_net.cuda()

    sequences_train = []
    sequences_valid = []
    for sid, subject in enumerate(dataset.subjects()):
        for action in dataset[subject].keys():
            if action.split('_')[1] == '1':
                # Only train on half the actions
                sequences_train.append((sid, subject, action))
            else:
                sequences_valid.append((sid, subject, action))

    print('Training on %d sequences, validating on %d sequences.' % (len(sequences_train), len(sequences_valid)))

    batch_size = 60
    n_epochs = 800
    train_subject_net(subject_net, batch_size, sequences_train, sequences_valid, dataset, n_epochs)

    # Save weights
    subject_weights_path = 'weights_subject.bin'
    subject_net.save_weights(subject_weights_path)
