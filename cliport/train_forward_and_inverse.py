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
from cliport.dataset import ForwardDatasetClassificationAllObjects
from cliport.utils import utils
from cliport.environments.environment import Environment
import wandb
from tqdm import tqdm
import clip
import numpy as np
import torchvision.models as models
import time

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

import sys
batch_size = int(sys.argv[1])
lr = float(sys.argv[2])

data_dir = os.path.join(root_dir, 'data')
train_dataset = ForwardDatasetClassificationAllObjects(os.path.join(data_dir, f'{cfg["task"]}-train'), cfg, n_demos=1000, augment=True)
train_data_loader = DataLoader(train_dataset, batch_size=batch_size)
# use train for now since val has different colors
test_dataset = ForwardDatasetClassificationAllObjects(os.path.join(data_dir, f'{cfg["task"]}-train'), cfg, n_demos=100, augment=True)
test_data_loader = DataLoader(test_dataset, batch_size=batch_size)

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


model = ICMModel().cuda()
mse = nn.MSELoss()
mse_no_reduction = nn.MSELoss(reduction='none')
ce = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
wandb.init(project='forward_inverse', entity="gnlp")

wandb.config.update({"exp_name": "forward_inverse",
                     "batch_size": batch_size,
                     "lr": lr})

wandb_dict = {"exp_name": "forward_inverse",
              "batch_size": batch_size,
              "lr": lr}

exp_name = ""

for (k,v) in wandb_dict.items():
    exp_name += "_{}-{}".format(k, v)

log_dir = '{}-{}'.format(time.strftime("%y-%m-%d-%H-%M-%S"), exp_name)
base_dir = os.path.join(root_dir, "data_log")

final_path = os.path.join(base_dir, log_dir)
if not os.path.exists(final_path):
    from pathlib import Path
    path = Path(final_path)
    path.mkdir(parents=True, exist_ok=True)

def train_or_val(flag, data_loader):
    if flag == 'train':
        model.train()
    else:
        model.eval()

    total = 0
    correct = 0
    for batch_idx, sample in enumerate(tqdm(data_loader)):
        state, next_state, action = sample
        state = state.cuda().float().permute(0,3,1,2)/255.
        next_state = next_state.cuda().float().permute(0,3,1,2)/255.
        action = action.long().cuda()
        # use one hot embedding for language in the forward model; can use clip embedding instead; should compare performance
        action_one_hot_embedding = nn.functional.one_hot(action, num_classes=len(all_languages))

        real_next_state_feature, pred_next_state_feature, pred_action = model(state, next_state, action_one_hot_embedding)
        inverse_loss = ce(pred_action, action)
        forward_loss = mse(pred_next_state_feature, real_next_state_feature.detach())

        if flag == 'train':
            loss = forward_loss + 0.05 * inverse_loss
            optimizer.zero_grad()
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

        predicted = torch.argmax(pred_action, 1)
        total += pred_action.shape[0]
        correct += (predicted == action).float().sum()
    return forward_loss, inverse_loss, correct / total

if TRAIN:
    for epoch in tqdm(range(1000)):
        print()
        print("Epoch: ", epoch)
        # train
        forward_loss_train, inverse_loss_train, inverse_accuracy_train = train_or_val('train', train_data_loader)
        # eval
        forward_loss_val, inverse_loss_val, inverse_accuracy_val = train_or_val('val', test_data_loader)

        wandb.log({"Train/forward_loss_train": forward_loss_train.item(),
                "Train/inverse_loss_train": inverse_loss_train.item(),
                "Val/forward_loss_val": forward_loss_val.item(),
                "Val/inverse_loss_val": inverse_loss_val.item(),
                "Train/inverse_acc_train": inverse_accuracy_train.item(),
                "Val/inverse_acc_val": inverse_accuracy_val.item()})
        
        if epoch % 10 == 0:
            torch.save(model,
                    os.path.join(final_path, "epoch_{}.pt".format(epoch)))
        
# evaluate 
# compose a sequence of different actions: different verbs/tasks, different nouns/pbjects. Use inverse model to segment the sequence.
# image-goal vs. no-goal vs. lang-goal vs. (oracle) image-goal
