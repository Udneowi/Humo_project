import torch.nn as nn
import torch.nn.functional as F
from pose_network_long_term import PoseNetworkLongTerm

class PoseNetworkLongTermSubject(PoseNetworkLongTerm):

    def __init__(self, prefix_length, skeleton, weight_path):
        super().__init__(prefix_length, skeleton)
        self.model.load_weights(weight_path)
        self.model.requires_grad = False

        self.num_input = self.model.num_joints*4
        self.fc1 = nn.Linear(num_input, 4*num_input)
        self.fc2 = nn.Linear(4*num_input, num_input)
        self.relu = nn.LeakyReLU(0.05, inplace=True)

    def forward(self, x, h=None, return_prenorm=False, return_all=False):
        x, h, x_pre = self.model(x, h, True, return_all)
        x_pre = self.relu(x_pre)
        x_pre = self.relu(self.fc1(x_pre))
        x_pre = self.fc2(x_pre)

        pre_normalized = x_pre
        normalized = F.normalize(x, dim=1).view(pre_normalized.shape)

        if self.num_outputs > 0:
            x = torch.cat((normalized, x[:, :, self.num_joints*4:]), dim=2)
        else:
            x = normalized

        if return_prenorm:
            return x, h, pre_normalized
        else:
            return x, h
