import numpy as np
import pdb

import roboverse.bullet as bullet
from roboverse.envs.sawyer_base import SawyerBaseEnv

class SawyerMultiSoupEnv(SawyerBaseEnv):

    def __init__(self, goal_pos=[.75,0,-.05], goal_obj='obj_0', *args, goal_mult=4, bonus=0, min_reward=-3., max_force=100., **kwargs):
        self.record_args(locals())
        self._goal_pos = goal_pos
        self._goal_obj = goal_obj
        super().__init__(*args, **kwargs)
        
        self._goal_mult = goal_mult
        self._bonus = bonus
        self._min_reward = min_reward
        self._id = 'SawyerLiftEnv'

    def get_goal_obj(self):
        return self._objects[self._goal_obj]

    def get_params(self):
        params = super().get_params()
        labels = ['_goal_pos', '_goal_obj', '_goal_mult', '_min_reward']
        params.update({label: getattr(self, label) for label in labels})
        return params

    def _load_meshes(self):
        super()._load_meshes()
        self._objects.update({
            'bowl':  bullet.objects.bowl(),
            'obj_0': bullet.objects.cube(pos=[.75, -.7, -.3]),
            'obj_1': bullet.objects.spam(pos=[.75, -.6, -.3]),
            'obj_2': bullet.objects.lego(pos=[.75, -.5, -.3]),
            'obj_3': bullet.objects.spam(pos=[.75, -.4, -.3]),

            'obj_4': bullet.objects.cube(pos=[.75, -.3, -.3]),
            'obj_5': bullet.objects.spam(pos=[.75, -.2, -.3]),
            'obj_6': bullet.objects.lego(pos=[.75, .35, -.3]),
            'obj_7': bullet.objects.spam(pos=[.75, .45, -.3]),
        })

        xyz_min = [.7, -.1, -.38]
        xyz_max = [.8, .1, -.35]
        visualize = True

        self._sensors.update({
            self._goal_obj: bullet.Sensor(self._objects[self._goal_obj], xyz_min=xyz_min, xyz_max=xyz_max, visualize=visualize)
        })

    def get_reward(self, observation):
        return self._sensors[self._goal_obj].sense()

    def get_termination(self, observation):
        return self._sensors[self._goal_obj].sense()

        # cube_pos = bullet.get_midpoint(self._objects['cube'])
        # ee_pos = bullet.get_link_state(self._sawyer, self._end_effector, 'pos')
        # ee_dist = bullet.l2_dist(cube_pos, ee_pos)
        # goal_dist = bullet.l2_dist(cube_pos, self._goal_pos)
        # reward = -(ee_dist + self._goal_mult * goal_dist)
        # reward = max(reward, self._min_reward)
        # if goal_dist < 0.25:
        #     reward += self._bonus
        # print(self._sensor_lid.sense(), self._sensor_cube.sense())
        # return reward

