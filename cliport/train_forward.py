"""Main training script."""

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
from cliport.dataset import ForwardDataset
from cliport.utils import utils
from cliport.environments.environment import Environment
import wandb
from tqdm import tqdm
import clip
import numpy as np


mode = 'train'
augment = True
TRAIN = True
task = 'packing-stacking-putting-same-objects'
# Load configs
root_dir = os.environ['CLIPORT_ROOT']
config_file = 'train.yaml' 
cfg = utils.load_hydra_config(os.path.join(root_dir, f'cliport/cfg/{config_file}'))

# Override defaults
cfg['task'] = task
cfg['mode'] = mode

data_dir = os.path.join(root_dir, 'data')
train_dataset = ForwardDataset(os.path.join(data_dir, f'{cfg["task"]}-train'), cfg, n_demos=2, augment=None)
train_data_loader = DataLoader(train_dataset, batch_size=2)
test_dataset = ForwardDataset(os.path.join(data_dir, f'{cfg["task"]}-val'), cfg, n_demos=2, augment=None)
test_data_loader = DataLoader(test_dataset, batch_size=2)


class ICMModel(nn.Module):
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

        '''
        self.feature = nn.Sequential(
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 512)
        )
        '''
        
        self.inverse_net = nn.Sequential(
            nn.Linear(512 * 2, 512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )

        self.residual = [nn.Sequential(
            nn.Linear(512 + 512, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
        ).to(self.device)] * 8

        self.forward_net_1 = nn.Sequential(
            nn.Linear(512 + 512, 512),
            nn.LeakyReLU()
        )
        self.forward_net_2 = nn.Sequential(
            nn.Linear(512 + 512, 512),
        )

        for p in self.modules():
            if isinstance(p, nn.Conv2d):
                init.kaiming_uniform_(p.weight)
                p.bias.data.zero_()

            if isinstance(p, nn.Linear):
                init.kaiming_uniform_(p.weight, a=1.0)
                p.bias.data.zero_()

    def forward(self, state, next_state, action):
        print(state.shape)
        encode_state = self.feature(state)
        encode_next_state = self.feature(next_state)
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


model = ICMModel().cuda()
mse = nn.MSELoss()
mse_no_reduction = nn.MSELoss(reduction='none')
optimizer = optim.Adam(model.parameters(), lr=1e-4)
wandb.init(project='forward_inverse_model')


all_languages = np.load(data_dir + "/language_dictionary.npy")
all_actions = np.load(data_dir + "/action_dictionary.npy")

def evaluate_inverse_model(ds, model):
    pred_languages_all_episode = []
    for i in range(len(ds)):
        episode, seed = ds.load(i)
        pred_languages = []

        for step in range(len(episode)-1):
            state = ds.process_sample(episode[step], augment=None)
            next_state = ds.process_sample(episode[step + 1], augment=None)
            image = torch.from_numpy(state['img'])[None,:,:,:3]
            next_image = torch.from_numpy(next_state['img'])[None,:,:,:3]
            image = image.cuda().float().permute(0,3,1,2)/255.
            next_image = next_image.cuda().float().permute(0,3,1,2)/255.
            # ground-truth lang goal
            lang_goal = state['lang_goal']
            with torch.no_grad():
                encode_state = model.feature(image)
                encode_next_state = model.feature(next_image)
                # get pred action
                pred_action = torch.cat((encode_state, encode_next_state), 1)
                pred_action = model.inverse_net(pred_action)
            action_idx = mse_no_reduction(pred_action, all_actions).mean(1).argmin()

            pred_language = all_languages[action_idx]
            pred_languages.append(pred_language)
        pred_languages_all_episode.append(pred_languages)
    return pred_languages_all_episode


def train_or_val(flag, data_loader):
    if flag == 'train':
        model.train()
    else:
        model.eval()

    for batch_idx, sample in enumerate(tqdm(data_loader)):
        state, next_state, action = sample
        state = state.cuda().float().permute(0,3,1,2)/255.
        next_state = next_state.cuda().float().permute(0,3,1,2)/255.
        action = action.float().cuda()
        real_next_state_feature, pred_next_state_feature, pred_action = model(state, next_state, action)
        forward_loss = mse(pred_next_state_feature, real_next_state_feature.detach())
        inverse_loss = mse(pred_action, action)
        if flag == 'train':
            loss = forward_loss + inverse_loss
            optimizer.zero_grad()
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
    return forward_loss, inverse_loss

if TRAIN:
    for epoch in tqdm(range(100)):
        print()
        print("Epoch: ", epoch)
        # train
        forward_loss_train, inverse_loss_train = train_or_val('train', train_data_loader)
        # eval
        forward_loss_val, inverse_loss_val = train_or_val('val', test_data_loader)
        pred_languages = evaluate_inverse_model(train_dataset, model)
        print(pred_languages)
        wandb.log({"forward_loss_train": forward_loss_train.item(),
                "inverse_loss_train": inverse_loss_train.item(),
                "forward_loss_val": forward_loss_val.item(),
                "inverse_loss_val": inverse_loss_val.item()})

        if epoch % 10 == 0:
            torch.save(model, "icm_model.pt")
else:
    model = torch.load("icm_model.pt")
    pred_languages = evaluate_inverse_model(train_dataset, model)
    print(pred_languages)
        
# evaluate 
# compose a sequence of different actions: different verbs/tasks, different nouns/pbjects. Use inverse model to segment the sequence.
# image-goal vs. no-goal vs. lang-goal vs. (oracle) image-goal
