"""Attention module."""

import numpy as np
import torch
import torch.nn.functional as F


from cliport.models.core.attention import Attention
from cliport.models.forward_model import ICMModel, EmbToImg
import torch.nn as nn

class AttentionImageGoal(Attention):
    """Attention (a.k.a Pick) with image-goals module."""

    def __init__(self, stream_fcn, in_shape, n_rotations, preprocess, cfg, device):
        super().__init__(stream_fcn, in_shape, n_rotations, preprocess, cfg, device)

    def forward(self, inp_img, goal_img, softmax=True):
        """Forward pass."""
        # Input image.
        in_data = np.pad(inp_img, self.padding, mode='constant')
        in_shape = (1,) + in_data.shape
        in_data = in_data.reshape(in_shape)
        in_tens = torch.from_numpy(in_data).to(dtype=torch.float, device=self.device)

        goal_tensor = np.pad(goal_img, self.padding, mode='constant')
        goal_shape = (1,) + goal_tensor.shape
        goal_tensor = goal_tensor.reshape(goal_shape)
        goal_tensor = torch.from_numpy(goal_tensor.copy()).to(dtype=torch.float, device=self.device)
        in_tens = in_tens * goal_tensor

        # Rotation pivot.
        pv = np.array(in_data.shape[1:3]) // 2

        # Rotate input.
        in_tens = in_tens.permute(0, 3, 1, 2)
        in_tens = in_tens.repeat(self.n_rotations, 1, 1, 1)
        in_tens = self.rotator(in_tens, pivot=pv)

        # Forward pass.
        logits = []
        for x in in_tens:
            logits.append(self.attend(x))
        logits = torch.cat(logits, dim=0)

        # Rotate back output.
        logits = self.rotator(logits, reverse=True, pivot=pv)
        logits = torch.cat(logits, dim=0)
        c0 = self.padding[:2, 0]
        c1 = c0 + inp_img.shape[:2]
        logits = logits[:, :, c0[0]:c1[0], c0[1]:c1[1]]

        logits = logits.permute(1, 2, 3, 0)  # D H W C
        output = logits.reshape(1, np.prod(logits.shape))
        if softmax:
            output = F.softmax(output, dim=-1)
            output = output.reshape(logits.shape[1:])
        return output


class AttentionEmbeddingGoal(Attention):
    """Attention (a.k.a Pick) with image-goals module."""

    def __init__(self, stream_fcn, in_shape, n_rotations, preprocess, cfg, device):
        super().__init__(stream_fcn, in_shape, n_rotations, preprocess, cfg, device)
        self.icm = ICMModel().cuda()
        self.icm.load_state_dict(torch.load("/home/jerrylin/temp/cliport/icm_model.pt"))
        self.icm.eval()
        self.all_languages = np.load('/home/jerrylin/temp/cliport/data/language_dictionary.npy')
        self.attention_emb2img = EmbToImg().cuda()
        self.attention_emb2img_optimizers = torch.optim.Adam(self.attention_emb2img.parameters(), lr=self.cfg['train']['lr'])

    def forward(self, inp_img, goal_img, lang_goal, softmax=True):
        """Forward pass."""
        # get goal embedding with forward model
        forward_model_cur_state = torch.from_numpy(inp_img[None]).cuda().float().permute(0,3,1,2)/255.
        forward_model_goal_state = torch.from_numpy(goal_img[None]).cuda().float().permute(0,3,1,2)/255.
        action = torch.from_numpy(lang_goal[None]).long().cuda()
        # use one hot embedding for language in the forward model; can use clip embedding instead; should compare performance
        action_one_hot_embedding = nn.functional.one_hot(action, num_classes=len(self.all_languages))
        real_next_state_feature, pred_next_state_feature, pred_action = self.icm(forward_model_cur_state, forward_model_goal_state, action_one_hot_embedding)
        goal_embedding = pred_next_state_feature.reshape((32,4,4))
        goal_tensor = self.attention_emb2img(goal_embedding[None])[0].permute(1,2,0)

        goal_tensor = nn.functional.pad(goal_tensor, (0,0,80,80,0,0), mode='constant')[None]
        # Input image.
        # Jerry: Need to check pad! Seems to be a bug!
        in_data = np.pad(inp_img, self.padding, mode='constant')
        in_shape = (1,) + in_data.shape
        in_data = in_data.reshape(in_shape)
        in_tens = torch.from_numpy(in_data).to(dtype=torch.float, device=self.device)
        in_tens = in_tens * goal_tensor

        # Rotation pivot.
        pv = np.array(in_data.shape[1:3]) // 2

        # Rotate input.
        in_tens = in_tens.permute(0, 3, 1, 2)
        in_tens = in_tens.repeat(self.n_rotations, 1, 1, 1)
        in_tens = self.rotator(in_tens, pivot=pv)

        # Forward pass.
        logits = []
        for x in in_tens:
            logits.append(self.attend(x))
        logits = torch.cat(logits, dim=0)

        # Rotate back output.
        logits = self.rotator(logits, reverse=True, pivot=pv)
        logits = torch.cat(logits, dim=0)
        c0 = self.padding[:2, 0]
        c1 = c0 + inp_img.shape[:2]
        logits = logits[:, :, c0[0]:c1[0], c0[1]:c1[1]]

        logits = logits.permute(1, 2, 3, 0)  # D H W C
        output = logits.reshape(1, np.prod(logits.shape))
        if softmax:
            output = F.softmax(output, dim=-1)
            output = output.reshape(logits.shape[1:])
        return output