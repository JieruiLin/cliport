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

def simple_feature(output_size=512):
    feature_output = 16 * 6 * 64
    return nn.Sequential(
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
        nn.Linear(feature_output, output_size)
    )

class ICMModel(nn.Module):
    def __init__(self,
                 encoder=simple_feature,
                 encoder_output_size=512,
                 use_cuda=True):

        super(ICMModel, self).__init__()
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        self.feature = encoder(output_size=encoder_output_size)
        self.inverse_net = nn.Sequential(
            nn.Linear(encoder_output_size * 2, 512),
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

    def forward(self, state, next_state):
        encode_state = self.feature(state)
        encode_next_state = self.feature(next_state)
        # get pred action
        pred_action = torch.cat((encode_state, encode_next_state), 1)
        pred_action = self.inverse_net(pred_action)

        return pred_action

class ICMModel_Contrastive(nn.Module):
    def __init__(self,
                 encoder=simple_feature,
                 encoder_output_size=512,
                 use_cuda=True):

        super(ICMModel, self).__init__()
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        self.feature = encoder(output_size=encoder_output_size)
        self.inverse_net = nn.Sequential(
            nn.Linear(encoder_output_size, 512),
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

    def forward_single(self, state):
        encode_state = self.feature(state)
        feature = self.inverse_net(encode_state)
        return feature

    def forward(self, state, next_state):
        state_feature = self.forward_single(state)
        next_state_feature = self.forward_single(next_state)
        return state_feature, next_state_feature

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on:
    """

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, x0, x1, y):
        # euclidian distance
        diff = x0 - x1
        dist_sq = torch.sum(torch.pow(diff, 2), 1)
        dist = torch.sqrt(dist_sq)

        mdist = self.margin - dist
        dist = torch.clamp(mdist, min=0.0)
        loss = y * dist_sq + (1 - y) * torch.pow(dist, 2)
        loss = torch.sum(loss) / 2.0 / x0.size()[0]
        return loss
