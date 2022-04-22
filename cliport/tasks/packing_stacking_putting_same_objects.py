"""Packing Shapes task."""

import os

import numpy as np
from cliport.tasks.task import Task
from cliport.tasks import primitives
from cliport.utils import utils
import random
import pybullet as p

class PackingStackingPuttingSameObjects(Task):

    def __init__(self):
        super().__init__()
        
        self.max_steps = 15
        self.train_set = np.arange(0, 14)
        self.test_set = np.arange(14, 20)
        self.homogeneous = False

        self.packing_lang_template = "pack the {obj} block in the brown box"
        self.packing_task_completed_desc = "done packing shapes."

        self.stacking_lang_template = "stack the {pick} block on {place}"
        self.stacking_task_completed_desc = "done stacking block pyramid."

        self.put_lang_template = "put the {pick} block in a {place} bowl"
        self.put_task_completed_desc = "done placing blocks in bowls."

        self.pos_eps = 0.2

    def reset(self, env):
        super().reset(env)

        # Add container box.
        zone_size = self.get_random_size(0.1, 0.15, 0.1, 0.15, 0.05, 0.05)
        zone_pose = self.get_random_pose(env, zone_size)
        container_template = 'container/container-template.urdf'
        half = np.float32(zone_size) / 2
        replace = {'DIM': zone_size, 'HALF': half}
        container_urdf = self.fill_template(container_template, replace)
        env.add_object(container_urdf, zone_pose, 'fixed')
        if os.path.exists(container_urdf):
            os.remove(container_urdf)

        # stacking
        # Add base.
        base_size = (0.05, 0.15, 0.005)
        base_urdf = 'stacking/stand.urdf'
        base_pose = self.get_random_pose(env, base_size)
        env.add_object(base_urdf, base_pose, 'fixed')
        
        # Block colors.
        import pdb 
        pdb.set_trace()
        color_names = self.get_colors()
        # Shuffle the block colors.
        random.shuffle(color_names)
        colors = [utils.COLORS[cn] for cn in color_names]
        
        # Add bowls.
        bowl_size = (0.12, 0.12, 0)
        bowl_urdf = 'bowl/bowl.urdf'
        bowl_poses = []
        for bc in range(2):
            bowl_pose = self.get_random_pose(env, bowl_size)
            bowl_id = env.add_object(bowl_urdf, bowl_pose, 'fixed')
            p.changeVisualShape(bowl_id, -1, rgbaColor=colors[bc] + [1])
            bowl_poses.append(bowl_pose)
        
        # Add blocks.
        objs = []
        block_size = (0.04, 0.04, 0.04)
        block_urdf = 'stacking/block.urdf'
        object_points = {}
        
        for i in range(7):
            block_pose = self.get_random_pose(env, block_size)
            block_id = env.add_object(block_urdf, block_pose)
            object_points[block_id] = self.get_box_object_points(block_id)
            p.changeVisualShape(block_id, -1, rgbaColor=colors[i] + [1])
            objs.append((block_id, (np.pi / 2, None)))

        # Associate placement locations for goals.
        place_pos = [(0, -0.05, 0.03), (0, 0, 0.03),
                     (0, 0.05, 0.03), (0, -0.025, 0.08),
                     (0, 0.025, 0.08), (0, 0, 0.13)]
        targs = [(utils.apply(base_pose, i), base_pose[1]) for i in place_pos]


        # Goal: make bottom row.
        self.goals.append(([objs[0]], np.ones((1, 1)), [targs[0]],
                           False, True, 'pose', None, 1 / 7))
        self.lang_goals.append(self.stacking_lang_template.format(pick=color_names[0],
                                                         place="the lightest brown block"))

        self.goals.append(([objs[1]], np.ones((1, 1)), [targs[1]],
                           False, True, 'pose', None, 1 / 7))
        self.lang_goals.append(self.stacking_lang_template.format(pick=color_names[1],
                                                         place="the middle brown block"))


        # insert one pack in the middle of stack
        obj_pts = dict()
        obj_pts[objs[4][0]] = object_points[objs[4][0]]
        self.goals.append(([objs[4]], np.int32([[1]]), [zone_pose],
                            False, True, 'zone',
                            (obj_pts, [(zone_pose, zone_size)]),
                            1 / 7))
        self.lang_goals.append(self.packing_lang_template.format(obj=color_names[4]))

        obj_pts = dict()
        obj_pts[objs[5][0]] = object_points[objs[5][0]]
        self.goals.append(([objs[5]], np.int32([[1]]), [zone_pose],
                            False, True, 'zone',
                            (obj_pts, [(zone_pose, zone_size)]),
                            1 / 7))
        self.lang_goals.append(self.packing_lang_template.format(obj=color_names[5]))


        self.goals.append(([objs[3]], np.ones((1, 1)), [targs[3]],
                           False, True, 'pose', None, 1 / 7))
        self.lang_goals.append(self.stacking_lang_template.format(pick=color_names[3],
                                                         place=f"the {color_names[0]} and {color_names[1]} blocks"))
        
        # Goal: put each block in a different bowl.
        self.goals.append(([objs[6]], np.ones((1, 1)),
                          [bowl_poses[0]], False, True, 'pose', None, 1 / 7))
        self.lang_goals.append(self.put_lang_template.format(pick=color_names[6],
                                                         place=color_names[0]))


        self.goals.append(([objs[2]], np.ones((1, 1)),
                           [bowl_poses[1]], False, True, 'pose', None, 1 / 7))
        self.lang_goals.append(self.put_lang_template.format(pick=color_names[2],
                                                         place=color_names[1]))
        
    def get_colors(self):
        return utils.TRAIN_COLORS if self.mode == 'train' else utils.EVAL_COLORS