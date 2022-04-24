import os
from pathlib import Path
from turtle import forward

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import init
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from cliport import agents
from cliport.dataset import ForwardDataset, ForwardDatasetClassification
from cliport.utils import utils
from cliport.environments.environment import Environment
import wandb
from tqdm import tqdm
import clip
import numpy as np


class ClassifyAction(nn.Module):
    def __init__(self, use_cuda=True):
        super(ICMModel, self).__init__()
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        feature_output = 16 * 6 * 64
        self.feature = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=8,
                stride=4),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=4,
                stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=4,
                stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(feature_output, 512)
        )

        self.inverse_net = nn.Sequential(
            nn.Linear(512 * 2, 512),
            nn.ReLU(),
            nn.Linear(512, 3)
        )

        for p in self.modules():
            if isinstance(p, nn.Conv2d):
                init.kaiming_uniform_(p.weight)
                p.bias.data.zero_()

            if isinstance(p, nn.Linear):
                init.kaiming_uniform_(p.weight, a=1.0)
                p.bias.data.zero_()

    def forward(self, state, next_state, action):
        encode_state = self.feature(state)
        encode_next_state = self.feature(next_state)
        # get pred action
        pred_action = torch.cat((encode_state, encode_next_state), 1)
        pred_action = self.inverse_net(pred_action)

        return pred_action