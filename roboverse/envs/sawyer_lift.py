import numpy as np
import pdb

import roboverse.bullet as bullet
from roboverse.envs.sawyer_base import SawyerBaseEnv

class SawyerLiftEnv(SawyerBaseEnv):

    def __init__(self, goal_pos=[.75,-.4,.2], *args, goal_mult=4, bonus=0, min_reward=-3., **kwargs):
        self.record_args(locals())
        super().__init__(*args, **kwargs)
        self._goal_pos = goal_pos
        self._goal_mult = goal_mult
        self._bonus = bonus
        self._min_reward = min_reward
        self._id = 'SawyerLiftEnv'

    def get_params(self):
        params = super().get_params()
        labels = ['_goal_pos', '_goal_mult', '_min_reward']
        params.update({label: getattr(self, label) for label in labels})
        return params

    def _load_meshes(self):
        super()._load_meshes()
        self._objects.update({
            'bowl':  bullet.objects.bowl(),
            'lid': bullet.objects.lid(),
            'cube': bullet.objects.spam()
        })

    def get_reward(self, observation):
        cube_pos = bullet.get_midpoint(self._objects['cube'])
        ee_pos = bullet.get_link_state(self._sawyer, self._end_effector, 'pos')
        ee_dist = bullet.l2_dist(cube_pos, ee_pos)
        goal_dist = bullet.l2_dist(cube_pos, self._goal_pos)
        reward = -(ee_dist + self._goal_mult * goal_dist)
        reward = max(reward, self._min_reward)
        if goal_dist < 0.25:
            reward += self._bonus
        # print(self._sensor_lid.sense(), self._sensor_cube.sense())
        return reward

