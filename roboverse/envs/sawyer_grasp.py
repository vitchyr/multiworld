import roboverse.bullet as bullet
from roboverse.envs.sawyer_base import SawyerBaseEnv


class SawyerGraspOneEnv(SawyerBaseEnv):

    def __init__(self, goal_pos=(.75,-.4,.2), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._goal_pos = goal_pos

    def _load_meshes(self):
        super()._load_meshes()
        self._objects = {
            'lego': bullet.objects.lego()
        }

    def get_reward(self, observation):
        object_pos = self.get_object_midpoint('lego')
        if object_pos[2] > -0.1:
            reward = 1
        else:
            reward = 0
        return reward
