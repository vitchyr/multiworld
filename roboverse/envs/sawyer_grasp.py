import numpy as np
import roboverse.bullet as bullet
from roboverse.envs.sawyer_base import SawyerBaseEnv


class SawyerGraspOneEnv(SawyerBaseEnv):

    def __init__(self, goal_pos=(.75,-.4,.2), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._goal_pos = goal_pos
        self._max_episode_steps = 200
        self._elapsed_steps = 0

    def _load_meshes(self):
        super()._load_meshes()
        self._objects = {
            'duck': bullet.objects.duck()
        }

    def get_reward(self, observation):
        object_pos = self.get_object_midpoint('duck')
        if object_pos[2] > -0.1:
            reward = 1
        else:
            reward = 0
        return reward

    def get_observation(self):
        return np.append(np.append(np.array(self.get_end_effector_pos()), np.array(self.get_object_midpoint('duck'))), np.array([self._current_gripper_target]))
