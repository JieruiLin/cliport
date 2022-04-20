"""Packing Shapes task."""

import os

import numpy as np
from cliport.tasks.task import Task
from cliport.utils import utils
import random
import pybullet as p

class PackingAndStacking(Task):

    def __init__(self):
        super().__init__()
        self.max_steps = 13
        self.train_set = np.arange(0, 14)
        self.test_set = np.arange(14, 20)
        self.homogeneous = False

        self.packing_lang_template = "pack the {obj} in the brown box"
        self.packing_task_completed_desc = "done packing shapes."

        self.stacking_lang_template = "put the {pick} block on {place}"
        self.stacking_task_completed_desc = "done stacking block pyramid."

    def reset(self, env):
        super().reset(env)

        # Shape Names:
        shapes = {
            0: "letter R shape",
            1: "letter A shape",
            2: "triangle",
            3: "square",
            4: "plus",
            5: "letter T shape",
            6: "diamond",
            7: "pentagon",
            8: "rectangle",
            9: "flower",
            10: "star",
            11: "circle",
            12: "letter G shape",
            13: "letter V shape",
            14: "letter E shape",
            15: "letter L shape",
            16: "ring",
            17: "hexagon",
            18: "heart",
            19: "letter M shape",
        }

        n_objects = 5
        if self.mode == 'train':
            obj_shapes = np.random.choice(self.train_set, n_objects, replace=False)
        else:
            if self.homogeneous:
                obj_shapes = [np.random.choice(self.test_set, replace=False)] * n_objects
            else:
                obj_shapes = np.random.choice(self.test_set, n_objects, replace=False)

        # Shuffle colors to avoid always picking an object of the same color
        color_names = self.get_colors()
        colors = [utils.COLORS[cn] for cn in color_names]
        np.random.shuffle(colors)

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

        # Add objects.
        objects = []
        template = 'kitting/object-template.urdf'
        object_points = {}
        for i in range(n_objects):
            shape = obj_shapes[i]
            size = (0.08, 0.08, 0.02)
            pose= self.get_random_pose(env, size)
            fname = f'{shape:02d}.obj'
            fname = os.path.join(self.assets_root, 'kitting', fname)
            scale = [0.003, 0.003, 0.001]  # .0005
            replace = {'FNAME': (fname,),
                       'SCALE': scale,
                       'COLOR': colors[i]}
            urdf = self.fill_template(template, replace)
            block_id = env.add_object(urdf, pose)
            if os.path.exists(urdf):
                os.remove(urdf)
            object_points[block_id] = self.get_box_object_points(block_id)
            objects.append((block_id, (0, None)))

        # Pick the first shape.
        num_objects_to_pick = 1
        for i in range(num_objects_to_pick):
            obj_pts = dict()
            obj_pts[objects[i][0]] = object_points[objects[i][0]]

            self.goals.append(([objects[i]], np.int32([[1]]), [zone_pose],
                               False, True, 'zone',
                               (obj_pts, [(zone_pose, zone_size)]),
                               1 / 7))
            self.lang_goals.append(self.packing_lang_template.format(obj=shapes[obj_shapes[i]]))

        # --------------------------------------------------------------------------
        # stacking
        # Add base.
        base_size = (0.05, 0.15, 0.005)
        base_urdf = 'stacking/stand.urdf'
        base_pose = self.get_random_pose(env, base_size)
        env.add_object(base_urdf, base_pose, 'fixed')

        # Block colors.
        color_names = self.get_colors()

        # Shuffle the block colors.
        random.shuffle(color_names)
        colors = [utils.COLORS[cn] for cn in color_names]

        # Add blocks.
        objs = []
        # sym = np.pi / 2
        block_size = (0.04, 0.04, 0.04)
        block_urdf = 'stacking/block.urdf'
        for i in range(6):
            block_pose = self.get_random_pose(env, block_size)
            block_id = env.add_object(block_urdf, block_pose)
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

        self.goals.append(([objs[2]], np.ones((1, 1)), [targs[2]],
                           False, True, 'pose', None, 1 / 7))
        self.lang_goals.append(self.stacking_lang_template.format(pick=color_names[2],
                                                         place="the darkest brown block"))

        # Goal: make middle row.
        self.goals.append(([objs[3]], np.ones((1, 1)), [targs[3]],
                           False, True, 'pose', None, 1 / 7))
        self.lang_goals.append(self.stacking_lang_template.format(pick=color_names[3],
                                                         place=f"the {color_names[0]} and {color_names[1]} blocks"))

        self.goals.append(([objs[4]], np.ones((1, 1)), [targs[4]],
                           False, True, 'pose', None, 1 / 7))
        self.lang_goals.append(self.stacking_lang_template.format(pick=color_names[4],
                                                         place=f"the {color_names[1]} and {color_names[2]} blocks"))

        # Goal: make top row.
        self.goals.append(([objs[5]], np.ones((1, 1)), [targs[5]],
                           False, True, 'pose', None, 1 / 7))
        self.lang_goals.append(self.stacking_lang_template.format(pick=color_names[5],
                                                         place=f"the {color_names[3]} and {color_names[4]} blocks"))

    def get_colors(self):
        return utils.TRAIN_COLORS if self.mode == 'train' else utils.EVAL_COLORS