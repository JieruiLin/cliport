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
from cliport.dataset import ForwardDataset, ForwardDatasetClassification, ForwardDatasetClassificationAllObjects
from cliport.utils import utils
from cliport.environments.environment import Environment
import wandb
from tqdm import tqdm
import clip
import numpy as np
import torchvision.models as models


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
BATCH_SIZE = int(sys.argv[1])

data_dir = os.path.join(root_dir, 'data')
train_dataset = ForwardDatasetClassificationAllObjects(os.path.join(data_dir, f'{cfg["task"]}-train'), cfg, n_demos=1000, augment=None)
train_data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
# use train set for val for now since not support other attributes
test_dataset = ForwardDatasetClassificationAllObjects(os.path.join(data_dir, f'{cfg["task"]}-train'), cfg, n_demos=100, augment=None)
test_data_loader = DataLoader(test_dataset, batch_size=8)

all_languages = np.load("/home/jerrylin/temp/cliport/data/language_dictionary.npy")
all_actions = np.load("/home/jerrylin/temp/cliport/data/action_dictionary.npy")

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

    def forward(self, state, next_state):
        encode_state = self.feature(state[:,:3])
        encode_next_state = self.feature(next_state[:,:3])
        # get pred action
        pred_action = torch.cat((encode_state, encode_next_state), 1)
        pred_action = self.inverse_net(pred_action)
       
        return pred_action


model = ICMModel().cuda()
mse = nn.MSELoss()
mse_no_reduction = nn.MSELoss(reduction='none')
ce = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
wandb.init(project='forward_inverse_model')
wandb.config.update({"batch_size": BATCH_SIZE})
def evaluate_inverse_model(ds, model):
    pred_languages_all_episode = []
    total = 0
    correct = 0
    model.eval()

    for i in range(10):
        episode_id = np.random.choice(ds.sample_set)
        episode, seed = ds.load(episode_id)
        pred_languages = []

        for step in range(len(episode)-1):
            _, _, reward, _, _ = episode[step + 1]
            if reward > 0:
                state = ds.process_sample(episode[step], augment=None)
                next_state = ds.process_sample(episode[step + 1], augment=None)
                image = torch.from_numpy(state['img'])[None]
                next_image = torch.from_numpy(next_state['img'])[None]
                image = image.cuda().float().permute(0,3,1,2)/255.
                next_image = next_image.cuda().float().permute(0,3,1,2)/255.
                # ground-truth lang goal
                lang_goal = state['lang_goal']
                cls = np.where(all_languages == lang_goal)[0]
                with torch.no_grad():
                    pred_action = model(image, next_image)

                predicted = torch.argmax(pred_action, 1)
                total += 1

                correct += float(predicted == cls)

                pred_language = all_languages[predicted]
                pred_languages.append(pred_language)
        pred_languages_all_episode.append(pred_languages)
    return pred_languages_all_episode, correct/total


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

        pred_action = model(state, next_state)
        inverse_loss = ce(pred_action, action)
        
        predicted = torch.argmax(pred_action, 1)
        total += pred_action.shape[0]
        correct += (predicted == action).float().sum()

        if flag == 'train':
            loss = inverse_loss
            optimizer.zero_grad()
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
    return inverse_loss, correct/total

if TRAIN:
    for epoch in tqdm(range(100)):
        print()
        print("Epoch: ", epoch)
        # train
        inverse_loss_train, accuracy_train = train_or_val('train', train_data_loader)
        # eval
        inverse_loss_val, accuracy_val = train_or_val('val', test_data_loader)
        pred_languages, accuracy_test = evaluate_inverse_model(train_dataset, model)
        print(pred_languages)
        
        wandb.log({"inverse_loss_train": inverse_loss_train.item(),
                   "inverse_loss_val": inverse_loss_val.item(),
                   "accuracy_train": accuracy_train.item(),
                   "accuracy_val": accuracy_val.item(),
                   "accuracy_test": accuracy_test})
        
        if epoch % 10 == 0:
            torch.save(model, "icm_model_all_classes_with_object_classes_bs{}_{}.pt".format(BATCH_SIZE, epoch))
else:
    pred_languages, accuracy = evaluate_inverse_model(train_dataset, model)
    print(accuracy)
    print(pred_languages)

        
# evaluate 
# compose a sequence of different actions: different verbs/tasks, different nouns/pbjects. Use inverse model to segment the sequence.
# image-goal vs. no-goal vs. lang-goal vs. (oracle) image-goal
