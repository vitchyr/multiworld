import numpy as np
import pdb

import roboverse.bullet as bullet
from roboverse.envs.sawyer_base import SawyerBaseEnv

class SawyerLiftEnv(SawyerBaseEnv):

    def __init__(self, goal_pos=[.75,-.4,.2], *args, goal_mult=4, min_reward=-3., **kwargs):
        self.record_args(locals())
        super().__init__(*args, **kwargs)
        self._goal_pos = goal_pos
        self._goal_mult = goal_mult
        self._min_reward = min_reward

    def get_params(self):
        params = super().get_params()
        labels = ['_goal_pos', '_goal_mult', '_min_reward']
        params.update({label: getattr(self, label) for label in labels})
        return params

    def _load_meshes(self):
        super()._load_meshes()
        self._objects = {
            'bowl':  bullet.objects.bowl(),
            'cube': bullet.objects.spam()
        }

    def get_reward(self, observation):
        # cube_pos = bullet.get_body_info(self._objects['cube'], 'pos')
        # l_finger_pos = bullet.get_link_state(self._sawyer, self._l_finger_tip, 'pos')
        # r_finger_pos = bullet.get_link_state(self._sawyer, self._r_finger_tip, 'pos')
        # ee_pos = (np.array(l_finger_pos) + np.array(r_finger_pos)) / 2.
        cube_pos = bullet.get_midpoint(self._objects['cube'])
        ee_pos = bullet.get_link_state(self._sawyer, self._gripper_site, 'pos')
        ee_dist = bullet.l2_dist(cube_pos, ee_pos)
        goal_dist = bullet.l2_dist(cube_pos, self._goal_pos)
        reward = -(ee_dist + self._goal_mult * goal_dist)
        reward = max(reward, self._min_reward)
        # print(self._sensor_lid.sense(), self._sensor_cube.sense())
        return reward

