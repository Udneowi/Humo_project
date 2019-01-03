import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from common.quaternet import QuaterNet
from time import time
from common.spline import Spline

class SubjectNet(nn.Module):
    def __init__(self, num_joints, num_subjects, prefix_length, qn_weight_path):
        super().__init__()

        self.prefix_length = prefix_length
        self.num_joints = num_joints
        self.num_subjects = num_subjects

        self.qn = QuaterNet(num_joints)
        self.qn.load_state_dict(torch.load(qn_weight_path, map_location=lambda storage, loc: storage))
        for p in self.qn.parameters():
            p.requires_grad = False

        qn_out = 1000
        fc1_out = 256
        fc2_out = num_subjects

        self.use_cuda = False

        self.model = nn.Sequential(
            nn.LeakyReLU(0.05),
            nn.Linear(qn_out, fc1_out),
            nn.BatchNorm1d(fc1_out),
            nn.ReLU(),
            nn.Linear(fc1_out, fc2_out),
            nn.BatchNorm1d(fc2_out),
        )


    def cuda(self):
        self.use_cuda = True
        self.qn.cuda()
        super().cuda()
        return self


    def save_weights(self, path):
        print('Saving weights to', path)
        torch.save(self.model.state_dict(), path)


    def load_weights(self, path):
        print('Loading weights from', path)
        self.model.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))


    def forward(self, x):
        _, h = self.qn(x)
        return self.model(h[1,:,:])


def prepare_next_batch(subject_net, pace_net, gen_net, batch_size, sequences, dataset, include_real=False):
    prefix_length = subject_net.prefix_length
    num_subjects = subject_net.num_subjects
    num_joints = subject_net.num_joints

    buffer_quat = torch.zeros((batch_size, prefix_length, num_joints*4), dtype=torch.float32)
    buffer_quat_gen = torch.zeros((batch_size, prefix_length, num_joints*4), dtype=torch.float32)
    buffer_subject = torch.zeros(batch_size, dtype=torch.int64)
    if gen_net.use_cuda:
        buffer_quat = buffer_quat.cuda()
        buffer_subject = buffer_subject.cuda()

    sequences = np.random.permutation(sequences)

    traj = np.array([[i,0] for i in range(prefix_length)])
    spline = Spline(traj, closed=False)
    spline = pace_net.predict(spline, average_speed=0.5)
    # Magic fix for direction
    spline.tracks['direction'][0][:,0] = 1
    spline.tracks['direction'][0][:,1] = 1

    batch_idx = 0
    for i, (sid, subject, action) in enumerate(sequences):
        # Pick a random chunk from each sequence
        start_idx = np.random.randint(0, dataset[subject][action]['rotations'].shape[0] - prefix_length + 1)
        end_idx = start_idx + prefix_length

        action_prefix = {}
        for key, arr in dataset[subject][action].items():
            action_prefix[key] = arr[start_idx:end_idx]
            if gen_net.use_cuda:
                action_prefix[key] = action_prefix[key]

        _, rotations = gen_net.generate_motion(spline, action_prefix, do_print=False)
        rotations = rotations.squeeze()[:prefix_length]
        buffer_quat[batch_idx] = torch.from_numpy(
            dataset[subject][action]['rotations'][start_idx:end_idx].reshape( \
            prefix_length, -1))
        buffer_quat_gen[batch_idx] = rotations.reshape(-1, num_joints*4)
        buffer_subject[batch_idx] = int(sid)

        batch_idx += 1
        if batch_idx == batch_size or i == len(sequences) - 1:
            if include_real:
                yield buffer_quat[:batch_idx], buffer_subject[:batch_idx]
            yield buffer_quat_gen[:batch_idx], buffer_subject[:batch_idx]
            batch_idx = 0


def train_subject_net(subject_net, pace_net, gen_net, batch_size, sequences_train, sequences_valid, dataset,\
    include_real=False, n_epochs=1000, benchmark_every=10, validate_every=10):
    np.random.seed(1337)

    batch_size_valid = 30

    lr = 0.001
    optimizer = optim.Adam(subject_net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    losses = []
    valid_losses = []
    print('Training for %d epochs' % (n_epochs))
    start_epoch = 0
    start_time = time()
    try:
        for epoch in range(n_epochs):
            subject_net.train()
            batch_loss = 0.0
            n_correct = 0
            N = 0
            # Train
            for batch_in, batch_out in prepare_next_batch(subject_net, pace_net, gen_net, batch_size, sequences_train, dataset, include_real):
                inputs = batch_in
                outputs = batch_out

                if subject_net.use_cuda:
                    inputs = inputs.cuda()
                    outputs = outputs.cuda()

                optimizer.zero_grad()

                clfs = subject_net(inputs)
                loss = criterion(clfs, outputs)

                loss.backward()

                optimizer.step()

                n_correct += torch.nonzero(F.softmax(clfs, dim=1).argmax(dim=1) == outputs).size(0)
                batch_loss += loss.item() * inputs.shape[0]
                N += inputs.shape[0]

            batch_loss /= N
            losses.append(batch_loss)

            print('[%d] loss: %.5f, acc: %.5f' % (epoch + 1, batch_loss, n_correct / N))

            # Validate
            if epoch > 0 and (epoch+1) % validate_every == 0:
                with torch.no_grad():
                    subject_net.eval()
                    valid_loss = 0
                    n_correct = 0
                    N = 0
                    for batch_in, batch_out in prepare_next_batch(subject_net, pace_net, gen_net, batch_size, sequences_valid, dataset):
                        inputs = batch_in
                        outputs = batch_out

                        if subject_net.use_cuda:
                            inputs = inputs.cuda()
                            outputs = outputs.cuda()

                        clfs = subject_net(inputs)
                        loss = criterion(clfs, outputs)
                        n_correct += torch.nonzero(F.softmax(clfs, dim=1).argmax(dim=1) == outputs).size(0)
                        valid_loss += loss.item() * inputs.shape[0]
                        N += inputs.shape[0]
                    valid_loss /= N
                    valid_losses.append(valid_loss)

                    print('Validation loss: %.5f, acc: %.5f' % (valid_loss, n_correct / N))

            if epoch > 0 and (epoch+1) % benchmark_every == 0:
                next_time = time()
                time_per_epoch = (next_time - start_time)/(epoch - start_epoch)
                print('Benchmark:', time_per_epoch, 's per epoch')
                start_time = next_time
                start_epoch = epoch
    except KeyboardInterrupt:
        print('Training aborted.')

    print('Done')
    return subject_net, losses
