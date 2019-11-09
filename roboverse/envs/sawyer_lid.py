import numpy as np
import pdb

import roboverse.bullet as bullet
from roboverse.envs.sawyer_lift import SawyerLiftEnv

class SawyerLidEnv(SawyerLiftEnv):

    def __init__(self, *args, min_reward=-3., **kwargs):
        super().__init__(*args, **kwargs)
        self._min_reward = min_reward

    def _load_meshes(self):
        super()._load_meshes()
        self._sensor_lid = bullet.Sensor(self._lid, xyz_min=[.6, .2, -.38], xyz_max=[.9, .5, -.35], visualize=True)
        self._sensor_cube = bullet.Sensor(self._cube, xyz_min=[.7, -.1, -.38], xyz_max=[.8, .1, -.35], visualize=True)
        self._goal_pos = self._sensor_lid.get_pos()
        self._goal_pos[-1] += 0.3

    def get_reward(self, observation):
        lid_pos = bullet.get_midpoint(self._lid)
        ee_pos = bullet.get_link_state(self._sawyer, self._end_effector, 'pos')
        ee_dist = bullet.l2_dist(lid_pos, ee_pos)

        lid_reward = self._sensor_lid.sense()

        reward = -ee_dist + lid_reward
        reward = max(reward, self._min_reward)
        return reward
        # return lid_reward

    def get_termination(self, observation):
        return self._sensor_lid.sense()




