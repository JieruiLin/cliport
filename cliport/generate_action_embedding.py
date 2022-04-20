import os
import pickle
import warnings

import numpy as np
import torch
import clip
from PIL import Image


# specific to packing shapes task
mode = 'test'
info_path = '/home/jerrylin/temp/cliport/data/packing-shapes-' + mode + '/info/'
image_path = '/home/jerrylin/temp/cliport/data/packing-shapes-' + mode + '/color/'
embedding_path = '/home/jerrylin/temp/cliport/data/packing-shapes-' + mode + '/embedding/'

device = "cuda" if torch.cuda.is_available() else "cpu"
# "ViT-B/32"
model, preprocess = clip.load("ViT-B/32", device=device)

for fname in sorted(os.listdir(info_path)):
    if os.path.exists(os.path.join(embedding_path, fname)):
        continue
    print(fname)
    info = pickle.load(open(os.path.join(info_path, fname), 'rb'))
    images = pickle.load(open(os.path.join(image_path, fname), 'rb'))
    pre_image = images[0][0]
    post_image = images[1][0]
    lang_goal = info[0]['lang_goal']
    token = clip.tokenize(lang_goal).to(device)

    with torch.no_grad():
        pre_image = preprocess(Image.fromarray(np.uint8(pre_image))).unsqueeze(0).to(device)
        pre_image_features = model.encode_image(pre_image)
        post_image = preprocess(Image.fromarray(np.uint8(post_image))).unsqueeze(0).to(device)
        post_image_features = model.encode_image(post_image)
        action_features = model.encode_text(token)

    with open(os.path.join(embedding_path, fname), 'wb') as f:
        features = torch.cat((pre_image_features, post_image_features, action_features), 1)
        pickle.dump(features.detach().cpu().numpy(), f)
