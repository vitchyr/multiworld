import numpy as np
import pdb

import roboverse.bullet as bullet
from roboverse.envs.sawyer_lift import SawyerLiftEnv

class SawyerLidEnv(SawyerLiftEnv):

    def __init__(self, *args, goal_pos=[0.75,.35,-.065], min_reward=-3., **kwargs):
        super().__init__(*args, **kwargs)
        self._min_reward = min_reward
        self._goal_pos = goal_pos
        self._id = 'SawyerLidEnv'

    def _load_meshes(self):
        super()._load_meshes()
        self._sensors.update({
            'lid': bullet.Sensor(self._objects['lid'], xyz_min=[.6, .2, -.38], xyz_max=[.9, .5, -.35], visualize=True),
            'cube': bullet.Sensor(self._objects['cube'], xyz_min=[.7, -.1, -.38], xyz_max=[.8, .1, -.35], visualize=True),
        })

    def get_reward(self, observation):
        lid_pos = bullet.get_midpoint(self._objects['lid'])
        ee_pos = bullet.get_link_state(self._sawyer, self._end_effector, 'pos')
        ee_dist = bullet.l2_dist(lid_pos, ee_pos)

        lid_reward = self._sensors['lid'].sense()

        reward = -ee_dist + lid_reward
        reward = max(reward, self._min_reward)
        return reward

    def get_termination(self, observation):
        return self._sensors['lid'].sense()
