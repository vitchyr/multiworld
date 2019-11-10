import numpy as np
import pdb

import roboverse.bullet as bullet
from roboverse.envs.sawyer_lift import SawyerLiftEnv

class SawyerSoupEnv(SawyerLiftEnv):

    def __init__(self, *args, min_reward=-3., **kwargs):
        super().__init__(*args, **kwargs)
        self._min_reward = min_reward
        self._id = 'SawyerSoupEnv'

    def _load_meshes(self):
        super()._load_meshes()
        self._sensor_lid = bullet.Sensor(self._lid, xyz_min=[.6, .2, -.38], xyz_max=[.9, .5, -.35], visualize=True)
        self._sensor_cube = bullet.Sensor(self._cube, xyz_min=[.7, -.1, -.38], xyz_max=[.8, .1, -.35], visualize=True)

    def get_reward(self, observation):
        lid_pos = bullet.get_midpoint(self._lid)
        ee_pos = bullet.get_link_state(self._sawyer, self._end_effector, 'pos')
        ee_dist = bullet.l2_dist(lid_pos, ee_pos)

        lid_reward = self._sensor_lid.sense()
        cube_reward = self._sensor_cube.sense()
        reward = lid_reward and cube_reward
        return reward

    def get_termination(self, observation):
        return self._sensor_lid.sense() and self._sensor_cube.sense()




