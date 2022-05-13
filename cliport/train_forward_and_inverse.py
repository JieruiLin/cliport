"""Main training script."""

import os
from pathlib import Path
from turtle import forward

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import init
from torch.utils.data import DataLoader
from cliport.dataset import ForwardDatasetClassificationAllObjects
from cliport.utils import utils
import wandb
from tqdm import tqdm
import clip
import numpy as np
import torchvision.models as models
from cliport.models.forward_model import ICMModel

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
train_dataset = ForwardDatasetClassificationAllObjects(os.path.join(data_dir, f'{cfg["task"]}-train'), cfg, n_demos=2, augment=None)
train_data_loader = DataLoader(train_dataset, batch_size=2)
# use train for now since val has different colors
test_dataset = ForwardDatasetClassificationAllObjects(os.path.join(data_dir, f'{cfg["task"]}-train'), cfg, n_demos=2, augment=None)
test_data_loader = DataLoader(test_dataset, batch_size=2)

all_languages = np.load(data_dir + "/language_dictionary.npy")
all_actions = np.load(data_dir + "/action_dictionary.npy")

mse = nn.MSELoss()
ce = nn.CrossEntropyLoss()

def train_or_val(flag, data_loader, model, optimizer):
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

def main():
    wandb.init(project='forward_inverse_model')
    model = ICMModel().cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    if TRAIN:
        for epoch in tqdm(range(1000)):
            print()
            print("Epoch: ", epoch)
            # train
            forward_loss_train, inverse_loss_train, inverse_accuracy_train = train_or_val('train', train_data_loader, model, optimizer=optimizer)
            # eval
            forward_loss_val, inverse_loss_val, inverse_accuracy_val = train_or_val('val', test_data_loader, model, optimizer=None)

            wandb.log({"forward_loss_train": forward_loss_train.item(),
                    "inverse_loss_train": inverse_loss_train.item(),
                    "forward_loss_val": forward_loss_val.item(),
                    "inverse_loss_val": inverse_loss_val.item(),
                    "inverse_acc_train": inverse_accuracy_train.item(),
                    "inverse_acc_val": inverse_accuracy_val.item()})

            if epoch % 10 == 0:
                torch.save(model.state_dict(), "icm_model.pt")
            
    # evaluate 
    # compose a sequence of different actions: different verbs/tasks, different nouns/pbjects. Use inverse model to segment the sequence.
    # image-goal vs. no-goal vs. lang-goal vs. (oracle) image-goal

if __name__ == "__main__":
    main()
