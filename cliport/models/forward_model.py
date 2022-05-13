import torch
import torch.nn as nn
import numpy as np
import os
import torchvision.models as models

root_dir = os.environ['CLIPORT_ROOT']
data_dir = os.path.join(root_dir, 'data')
all_languages = np.load(data_dir + "/language_dictionary.npy")
all_actions = np.load(data_dir + "/action_dictionary.npy")

class ICMModel(nn.Module):
    def __init__(self, use_cuda=True):
        super(ICMModel, self).__init__()
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        
        self.feature = models.resnet18()
        self.feature.fc = nn.Linear(512, 512)

        self.inverse_net = nn.Sequential(
            nn.Linear(512 * 2, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, len(all_languages))
        )

        self.residual = [nn.Sequential(
            nn.Linear(512 + len(all_languages), 512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
        ).to(self.device)] * 8

        self.forward_net_1 = nn.Sequential(
            nn.Linear(512 + len(all_languages), 512),
            nn.LeakyReLU()
        )
        self.forward_net_2 = nn.Sequential(
            nn.Linear(512 + len(all_languages), 512),
        )

    def forward(self, state, next_state, action):
        encode_state = self.feature(state[:,:3])
        encode_next_state = self.feature(next_state[:,:3])
        # get pred action
        pred_action = torch.cat((encode_state, encode_next_state), 1)
        pred_action = self.inverse_net(pred_action)
        # ---------------------

        # get pred next state
        pred_next_state_feature_orig = torch.cat((encode_state, action), 1)
        pred_next_state_feature_orig = self.forward_net_1(pred_next_state_feature_orig)

        # residual
        for i in range(4):
            pred_next_state_feature = self.residual[i * 2](torch.cat((pred_next_state_feature_orig, action), 1))
            pred_next_state_feature_orig = self.residual[i * 2 + 1](
                torch.cat((pred_next_state_feature, action), 1)) + pred_next_state_feature_orig

        pred_next_state_feature = self.forward_net_2(torch.cat((pred_next_state_feature_orig, action), 1))

        real_next_state_feature = encode_next_state
        return real_next_state_feature, pred_next_state_feature, pred_action

class EmbToImg(nn.Module):
    def __init__(self, use_cuda=True):
        super(EmbToImg, self).__init__()
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        self.main = nn.Sequential(
            nn.ConvTranspose2d(32, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 6, 4, 2, 1, bias=False),
            nn.Upsample(size=(320,160), mode='bilinear')
        )

    def forward(self, embedding):
        return self.main(embedding)