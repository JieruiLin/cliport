import os
import pickle
import warnings

import numpy as np
import torch
import clip
from PIL import Image


mode = 'train'
task = 'packing-stacking-putting-same-objects'
info_path = '/home/jerrylin/temp/cliport/data/' + task + '-' + mode + '/info/'
embedding_path = '/home/jerrylin/temp/cliport/data/' + task + '-' + mode + '/embedding/'
if not os.path.exists(embedding_path):
    os.mkdir(embedding_path)

device = "cuda" if torch.cuda.is_available() else "cpu"
# "ViT-B/32"
model, preprocess = clip.load("ViT-B/32", device=device)

for fname in sorted(os.listdir(info_path)):
    if os.path.exists(os.path.join(embedding_path, fname)):
        continue
    print(fname)
    info = pickle.load(open(os.path.join(info_path, fname), 'rb'))

    # the last language goal is not used 
    lang_goals = []
    for i in range(len(info)-1):
        lang_goal = info[i]['lang_goal']
        lang_goals.append(lang_goal)
    token = clip.tokenize(lang_goals).to(device)

    with torch.no_grad():
        action_features = model.encode_text(token)

    with open(os.path.join(embedding_path, fname), 'wb') as f:
        pickle.dump(action_features.detach().cpu().numpy(), f)
