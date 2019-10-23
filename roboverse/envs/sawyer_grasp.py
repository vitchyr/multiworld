import roboverse.bullet as bullet
from roboverse.envs.sawyer_base import SawyerBaseEnv


class SawyerGraspOneEnv(SawyerBaseEnv):

    def __init__(self, goal_pos=(.75,-.4,.2), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._goal_pos = goal_pos

    def _load_meshes(self):
        super()._load_meshes()
        self._objects = {
            'duck': bullet.objects.duck()
        }

    def get_reward(self, observation):
        cube_pos = self.get_object_midpoint('duck')
        ee_pos = self.get_end_effector_pos()
        ee_dist = bullet.l2_dist(cube_pos, ee_pos)
        goal_dist = bullet.l2_dist(cube_pos, self._goal_pos)
        reward = -(ee_dist + 2*goal_dist)
        return reward
