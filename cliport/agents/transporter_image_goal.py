import numpy as np
from cliport.utils import utils
from cliport.agents.transporter import OriginalTransporterAgent
from cliport.models.core.attention import Attention
from cliport.models.core.attention_image_goal import AttentionImageGoal, AttentionEmbeddingGoal
from cliport.models.core.transport_image_goal import TransportImageGoal, TransportEmbeddingGoal
from cliport.models.forward_model import ICMModel
import torch
import torch.nn as nn

class ImageGoalTransporterAgent(OriginalTransporterAgent):
    def __init__(self, name, cfg, train_ds, test_ds):
        super().__init__(name, cfg, train_ds, test_ds)

    def _build_model(self):
        stream_fcn = 'plain_resnet'
        self.attention = AttentionImageGoal(
            stream_fcn=(stream_fcn, None),
            in_shape=self.in_shape,
            n_rotations=1,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )
        self.transport = TransportImageGoal(
            stream_fcn=(stream_fcn, None),
            in_shape=self.in_shape,
            n_rotations=self.n_rotations,
            crop_size=self.crop_size,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )

    def attn_forward(self, inp, softmax=True):
        inp_img = inp['inp_img']
        goal_img = inp['goal_img']
        out = self.attention.forward(inp_img, goal_img, softmax=softmax)
        return out

    def attn_training_step(self, frame, goal, backprop=True, compute_err=False):
        inp_img = frame['img']
        goal_img = goal['img']
        p0, p0_theta = frame['p0'], frame['p0_theta']

        inp = {'inp_img': inp_img, 'goal_img': goal_img}
        out = self.attn_forward(inp, softmax=False)
        return self.attn_criterion(backprop, compute_err, inp, out, p0, p0_theta)

    def trans_forward(self, inp, softmax=True):
        inp_img = inp['inp_img']
        goal_img = inp['goal_img']
        p0 = inp['p0']

        out = self.transport.forward(inp_img, goal_img, p0, softmax=softmax)
        return out

    def transport_training_step(self, frame, goal, backprop=True, compute_err=False):
        inp_img = frame['img']
        goal_img = goal['img']
        p0 = frame['p0']
        p1, p1_theta = frame['p1'], frame['p1_theta']

        inp = {'inp_img': inp_img, 'goal_img': goal_img, 'p0': p0}
        out = self.trans_forward(inp, softmax=False)
        err, loss = self.transport_criterion(backprop, compute_err, inp, out, p0, p1, p1_theta)
        return loss, err

    def training_step(self, batch, batch_idx):
        self.attention.train()
        self.transport.train()

        frame, goal = batch

        # Get training losses.
        step = self.total_steps + 1
        loss0, err0 = self.attn_training_step(frame, goal)
        if isinstance(self.transport, Attention):
            loss1, err1 = self.attn_training_step(frame, goal)
        else:
            loss1, err1 = self.transport_training_step(frame, goal)
        total_loss = loss0 + loss1
        self.log('tr/attn/loss', loss0)
        self.log('tr/trans/loss', loss1)
        self.log('tr/loss', total_loss)
        self.total_steps = step

        self.trainer.train_loop.running_loss.append(total_loss)

        self.check_save_iteration()

        return dict(
            loss=total_loss,
        )

    def validation_step(self, batch, batch_idx):
        self.attention.eval()
        self.transport.eval()

        loss0, loss1 = 0, 0
        for i in range(self.val_repeats):
            frame, goal = batch
            l0, err0 = self.attn_training_step(frame, goal, backprop=False, compute_err=True)
            loss0 += l0
            if isinstance(self.transport, Attention):
                l1, err1 = self.attn_training_step(frame, goal, backprop=False, compute_err=True)
                loss1 += l1
            else:
                l1, err1 = self.transport_training_step(frame, goal, backprop=False, compute_err=True)
                loss1 += l1
        loss0 /= self.val_repeats
        loss1 /= self.val_repeats
        val_total_loss = loss0 + loss1

        self.trainer.evaluation_loop.trainer.train_loop.running_loss.append(val_total_loss)

        return dict(
            val_loss=val_total_loss,
            val_loss0=loss0,
            val_loss1=loss1,
            val_attn_dist_err=err0['dist'],
            val_attn_theta_err=err0['theta'],
            val_trans_dist_err=err1['dist'],
            val_trans_theta_err=err1['theta'],
        )

    def act(self, obs, info=None, goal=None):  # pylint: disable=unused-argument
        """Run inference and return best action given visual observations."""
        # Get heightmap from RGB-D images.
        img = self.test_ds.get_image(obs)
        goal_img = self.test_ds.get_image(goal[0])

        # Attention model forward pass.
        pick_conf = self.attention.forward(img, goal_img)
        pick_conf = pick_conf.detach().cpu().numpy()
        argmax = np.argmax(pick_conf)
        argmax = np.unravel_index(argmax, shape=pick_conf.shape)
        p0_pix = argmax[:2]
        p0_theta = argmax[2] * (2 * np.pi / pick_conf.shape[2])

        # Transport model forward pass.
        place_conf = self.transport.forward(img, goal_img, p0_pix)
        place_conf = place_conf.permute(1, 2, 0)
        place_conf = place_conf.detach().cpu().numpy()
        argmax = np.argmax(place_conf)
        argmax = np.unravel_index(argmax, shape=place_conf.shape)
        p1_pix = argmax[:2]
        p1_theta = argmax[2] * (2 * np.pi / place_conf.shape[2])

        # Pixels to end effector poses.
        hmap = img[:, :, 3]
        p0_xyz = utils.pix_to_xyz(p0_pix, hmap, self.bounds, self.pix_size)
        p1_xyz = utils.pix_to_xyz(p1_pix, hmap, self.bounds, self.pix_size)
        p0_xyzw = utils.eulerXYZ_to_quatXYZW((0, 0, -p0_theta))
        p1_xyzw = utils.eulerXYZ_to_quatXYZW((0, 0, -p1_theta))

        return {
            'pose0': (np.asarray(p0_xyz), np.asarray(p0_xyzw)),
            'pose1': (np.asarray(p1_xyz), np.asarray(p1_xyzw)),
            'pick': p0_pix,
            'place': p1_pix,
        }


class EmbeddingGoalTransporterAgent(OriginalTransporterAgent):
    def __init__(self, name, cfg, train_ds, test_ds):
        super().__init__(name, cfg, train_ds, test_ds)
        self.all_languages = np.load('/home/jerrylin/temp/cliport/data/language_dictionary.npy')
        self.icm = ICMModel().cuda()
        self.icm.load_state_dict(torch.load("/home/jerrylin/temp/cliport/icm_model.pt"))
        self.icm.eval()

    def _build_model(self):
        stream_fcn = 'plain_resnet'
        self.attention = AttentionEmbeddingGoal(
            stream_fcn=(stream_fcn, None),
            in_shape=self.in_shape,
            n_rotations=1,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )
        self.transport = TransportEmbeddingGoal(
            stream_fcn=(stream_fcn, None),
            in_shape=self.in_shape,
            n_rotations=self.n_rotations,
            crop_size=self.crop_size,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )

    def attn_forward(self, inp, softmax=True):
        inp_img = inp['inp_img']
        goal_img = inp['goal_img']
        out = self.attention.forward(inp_img, goal_img, softmax=softmax)
        return out

    def attn_training_step(self, frame, goal, backprop=True, compute_err=False):
        inp_img = frame['img']

        p0, p0_theta = frame['p0'], frame['p0_theta']

        inp = {'inp_img': inp_img, 'goal_img': goal}
        out = self.attn_forward(inp, softmax=False)
        return self.attn_criterion(backprop, compute_err, inp, out, p0, p0_theta)

    def trans_forward(self, inp, softmax=True):
        inp_img = inp['inp_img']
        goal_img = inp['goal_img']

        p0 = inp['p0']

        out = self.transport.forward(inp_img, goal_img, p0, softmax=softmax)
        return out

    def transport_training_step(self, frame, goal, backprop=True, compute_err=False):
        inp_img = frame['img']

        p0 = frame['p0']
        p1, p1_theta = frame['p1'], frame['p1_theta']
        
        inp = {'inp_img': inp_img, 'goal_img': goal, 'p0': p0}
        out = self.trans_forward(inp, softmax=False)
        err, loss = self.transport_criterion(backprop, compute_err, inp, out, p0, p1, p1_theta)
        return loss, err

    def training_step(self, batch, batch_idx):
        self.attention.train()
        self.transport.train()

        frame, goal = batch

        # Get training losses.
        step = self.total_steps + 1

        # get goal embedding with forward model
        inp_img = frame['img']
        goal_img = goal['img']
        lang_goal = frame['lang_goal']
        lang_goal = np.where(self.all_languages == lang_goal)[0][0]

        forward_model_cur_state = torch.from_numpy(inp_img[None]).cuda().float().permute(0,3,1,2)/255.
        forward_model_goal_state = torch.from_numpy(goal_img[None]).cuda().float().permute(0,3,1,2)/255.
        action = torch.from_numpy(lang_goal[None]).long().cuda()
        # use one hot embedding for language in the forward model; can use clip embedding instead; should compare performance
        action_one_hot_embedding = nn.functional.one_hot(action, num_classes=len(self.all_languages))
        _, pred_next_state_feature, _ = self.icm(forward_model_cur_state, forward_model_goal_state, action_one_hot_embedding)
        # detach gradient
        goal_embedding = pred_next_state_feature.reshape((32,4,4)).detach()

        loss0, err0 = self.attn_training_step(frame, goal_embedding)
        if isinstance(self.transport, Attention):
            loss1, err1 = self.attn_training_step(frame, goal_embedding)
        else:
            loss1, err1 = self.transport_training_step(frame, goal_embedding)
        total_loss = loss0 + loss1
        self.log('tr/attn/loss', loss0)
        self.log('tr/trans/loss', loss1)
        self.log('tr/loss', total_loss)
        self.total_steps = step

        self.trainer.train_loop.running_loss.append(total_loss)

        self.check_save_iteration()

        return dict(
            loss=total_loss,
        )

    def validation_step(self, batch, batch_idx):
        self.attention.eval()
        self.transport.eval()

        loss0, loss1 = 0, 0
        for i in range(self.val_repeats):
            frame, goal = batch

            # get goal embedding with forward model
            inp_img = frame['img']
            goal_img = goal['img']
            lang_goal = frame['lang_goal']
            lang_goal = np.where(self.all_languages == lang_goal)[0][0]

            forward_model_cur_state = torch.from_numpy(inp_img[None]).cuda().float().permute(0,3,1,2)/255.
            forward_model_goal_state = torch.from_numpy(goal_img[None]).cuda().float().permute(0,3,1,2)/255.
            action = torch.from_numpy(lang_goal[None]).long().cuda()
            # use one hot embedding for language in the forward model; can use clip embedding instead; should compare performance
            action_one_hot_embedding = nn.functional.one_hot(action, num_classes=len(self.all_languages))
            _, pred_next_state_feature, _ = self.icm(forward_model_cur_state, forward_model_goal_state, action_one_hot_embedding)
            goal_embedding = pred_next_state_feature.reshape((32,4,4))

            l0, err0 = self.attn_training_step(frame, goal_embedding, backprop=False, compute_err=True)
            loss0 += l0
            if isinstance(self.transport, Attention):
                l1, err1 = self.attn_training_step(frame, goal_embedding, backprop=False, compute_err=True)
                loss1 += l1
            else:
                l1, err1 = self.transport_training_step(frame, goal_embedding, backprop=False, compute_err=True)
                loss1 += l1
        loss0 /= self.val_repeats
        loss1 /= self.val_repeats
        val_total_loss = loss0 + loss1

        self.trainer.evaluation_loop.trainer.train_loop.running_loss.append(val_total_loss)

        return dict(
            val_loss=val_total_loss,
            val_loss0=loss0,
            val_loss1=loss1,
            val_attn_dist_err=err0['dist'],
            val_attn_theta_err=err0['theta'],
            val_trans_dist_err=err1['dist'],
            val_trans_theta_err=err1['theta'],
        )

    def act(self, obs, info=None, goal=None):  # pylint: disable=unused-argument
        """Run inference and return best action given visual observations."""
        # Get heightmap from RGB-D images.
        # need to get goal image embedding
        img = self.test_ds.get_image(obs)
        lang_goal = info['lang_goal']
        goal_img = self.test_ds.get_image(goal[0])

        lang_goal = np.where(self.all_languages == lang_goal)[0][0]

        forward_model_cur_state = torch.from_numpy(img[None]).cuda().float().permute(0,3,1,2)/255.
        forward_model_goal_state = torch.from_numpy(goal_img[None]).cuda().float().permute(0,3,1,2)/255.
        action = torch.from_numpy(lang_goal[None]).long().cuda()
        # use one hot embedding for language in the forward model; can use clip embedding instead; should compare performance
        action_one_hot_embedding = nn.functional.one_hot(action, num_classes=len(self.all_languages))
        _, pred_next_state_feature, _ = self.icm(forward_model_cur_state, forward_model_goal_state, action_one_hot_embedding)
        goal_img = pred_next_state_feature.reshape((32,4,4))

        # Attention model forward pass.
        pick_conf = self.attention.forward(img, goal_img)
        pick_conf = pick_conf.detach().cpu().numpy()
        argmax = np.argmax(pick_conf)
        argmax = np.unravel_index(argmax, shape=pick_conf.shape)
        p0_pix = argmax[:2]
        p0_theta = argmax[2] * (2 * np.pi / pick_conf.shape[2])

        # Transport model forward pass.
        place_conf = self.transport.forward(img, goal_img, p0_pix)
        place_conf = place_conf.permute(1, 2, 0)
        place_conf = place_conf.detach().cpu().numpy()
        argmax = np.argmax(place_conf)
        argmax = np.unravel_index(argmax, shape=place_conf.shape)
        p1_pix = argmax[:2]
        p1_theta = argmax[2] * (2 * np.pi / place_conf.shape[2])

        # Pixels to end effector poses.
        hmap = img[:, :, 3]
        p0_xyz = utils.pix_to_xyz(p0_pix, hmap, self.bounds, self.pix_size)
        p1_xyz = utils.pix_to_xyz(p1_pix, hmap, self.bounds, self.pix_size)
        p0_xyzw = utils.eulerXYZ_to_quatXYZW((0, 0, -p0_theta))
        p1_xyzw = utils.eulerXYZ_to_quatXYZW((0, 0, -p1_theta))

        return {
            'pose0': (np.asarray(p0_xyz), np.asarray(p0_xyzw)),
            'pose1': (np.asarray(p1_xyz), np.asarray(p1_xyzw)),
            'pick': p0_pix,
            'place': p1_pix,
        }

