import argparse
import gym
import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import hydra
from cliport import agents
from cliport import dataset
from cliport import tasks
from cliport.utils import utils
from cliport.tasks import cameras
from cliport.environments.environment import Environment
from cliport.models.streams.one_stream_attention_lang_fusion import OneStreamAttentionLangFusion
from cliport.models.streams.one_stream_transport_lang_fusion import OneStreamTransportLangFusion


parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()


env = Environment(
        '/home/jlin/cliport/cliport/environments/assets/',
        disp=False,
        shared_memory=False,
        hz=480
    )
task_name = 'packing-shapes'
in_shape = (320, 160, 6)
pix_size = 0.003125
bounds = np.array([[0.25, 0.75], [-0.5, 0.5], [0, 0.28]])
task = tasks.names[task_name]()
task.mode = 'train'
env.seed(0)
env.set_task(task)
obs = env.reset()
info = env.info
torch.manual_seed(0)


eps = np.finfo(np.float32).eps.item()
cam_config = cameras.RealSenseD415.CONFIG


def finish_episode(episode_rewards, episode_log_probs, _optimizers):
    import pdb 
    pdb.set_trace()
    R = 0
    policy_loss = []
    returns = []
    for r in episode_rewards[::-1]:
        R = r + args.gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)
    for log_prob, R in zip(episode_log_probs, returns):
        policy_loss.append(-log_prob * R)
    _optimizers['attn'].zero_grad()
    _optimizers['trans'].zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    _optimizers['attn'].step()
    _optimizers['trans'].step()

def get_image(obs):
    """Stack color and height images image."""

    # Get color and height maps from RGB-D images.
    cmap, hmap = utils.get_fused_heightmap(
        obs, cam_config, bounds, pix_size)
    img = np.concatenate((cmap,
                            hmap[Ellipsis, None],
                            hmap[Ellipsis, None],
                            hmap[Ellipsis, None]), axis=2)
    assert img.shape == in_shape, img.shape
    return img


def attn_forward(attention, inp, softmax=True):
    inp_img = inp['inp_img']
    lang_goal = inp['lang_goal']

    out = attention.forward(inp_img, lang_goal, softmax=softmax)
    return out

def trans_forward(transport, inp, softmax=True):
    inp_img = inp['inp_img']
    p0 = inp['p0']
    lang_goal = inp['lang_goal']

    out = transport.forward(inp_img, p0, lang_goal, softmax=softmax)
    return out

def act(attention, transport, obs, info, goal=None):  # pylint: disable=unused-argument
    """Run inference and return best action given visual observations."""
    # Get heightmap from RGB-D images.
    img = get_image(obs)
    lang_goal = info['lang_goal']
    # Attention model forward pass.
    pick_inp = {'inp_img': img, 'lang_goal': lang_goal}
    pick_conf = attn_forward(attention, pick_inp)
    m = Categorical(pick_conf.flatten())
    import pdb 
    pdb.set_trace()
    p0_idx = m.sample()
    p0_log_prob = m.log_prob(p0_idx)
    p0_idx = p0_idx.detach().cpu().numpy()
    p0_idx = np.unravel_index(p0_idx, shape=pick_conf.shape)
    p0_pix = p0_idx[:2]
    p0_theta = p0_idx[2] * (2 * np.pi / pick_conf.shape[2])

    # Transport model forward pass.
    place_inp = {'inp_img': img, 'p0': p0_pix, 'lang_goal': lang_goal}
    place_conf = trans_forward(transport, place_inp)
    place_conf = place_conf.permute(1, 2, 0)
    m = Categorical(place_conf.flatten())
    p1_idx = m.sample()
    p1_log_prob = m.log_prob(p1_idx)
    p1_idx = p1_idx.detach().cpu().numpy()
    p1_idx = np.unravel_index(p1_idx, shape=place_conf.shape)
    p1_pix = p1_idx[:2]
    p1_theta = p1_idx[2] * (2 * np.pi / place_conf.shape[2])
    # Pixels to end effector poses.
    hmap = img[:, :, 3]
    p0_xyz = utils.pix_to_xyz(p0_pix, hmap, bounds, pix_size)
    p1_xyz = utils.pix_to_xyz(p1_pix, hmap, bounds, pix_size)
    p0_xyzw = utils.eulerXYZ_to_quatXYZW((0, 0, -p0_theta))
    p1_xyzw = utils.eulerXYZ_to_quatXYZW((0, 0, -p1_theta))

    return {
        'pose0': (np.asarray(p0_xyz), np.asarray(p0_xyzw)),
        'pose1': (np.asarray(p1_xyz), np.asarray(p1_xyzw)),
        'pick': [p0_pix[0], p0_pix[1], p0_theta],
        'place': [p1_pix[0], p1_pix[1], p1_theta],
    }, p0_log_prob + p1_log_prob

@hydra.main(config_path="./cfg", config_name='train')
def main(cfg):
    running_reward = 10
    stream_fcn = 'clip_lingunet'
    device_type = torch.device('cuda')
    attention = OneStreamAttentionLangFusion(
        stream_fcn=(stream_fcn, None),
        in_shape=in_shape,
        n_rotations=1,
        preprocess=utils.preprocess,
        cfg=cfg,
        device=device_type,
    )
    transport = OneStreamTransportLangFusion(
        stream_fcn=(stream_fcn, None),
        in_shape=in_shape,
        n_rotations=cfg['train']['n_rotations'],
        crop_size=64,
        preprocess=utils.preprocess,
        cfg=cfg,
        device=device_type,
    )

    _optimizers = {
        'attn': torch.optim.Adam(attention.parameters(), lr=cfg['train']['lr']),
        'trans': torch.optim.Adam(transport.parameters(), lr=cfg['train']['lr'])
    }

    for i_episode in count(1):
        obs, ep_reward = env.reset(), 0
        info = env.info
        episode_rewards = []
        episode_log_probs = []
        for t in range(1, 10000):  # Don't infinite loop while learning
            print(t)
            action, action_log_prob = act(attention, transport, obs, info, goal=None)
            obs, reward, done, info = env.step(action)
            import pdb 
            pdb.set_trace()
            episode_rewards.append(reward)
            episode_log_probs.append(action_log_prob)
            ep_reward += reward
            if done:
                break
        import pdb 
        pdb.set_trace()
        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        finish_episode(episode_rewards, episode_log_probs, _optimizers)
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  i_episode, ep_reward, running_reward))
        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break


if __name__ == '__main__':
    main()