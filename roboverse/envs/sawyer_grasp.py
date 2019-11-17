import roboverse.bullet as bullet
import numpy as np
from roboverse.envs.sawyer_base import SawyerBaseEnv
import pybullet

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

    def _format_state_query(self):
        ## position and orientation of body root
        bodies = [v for k,v in self._objects.items()]
        ## position and orientation of link
        links = [(self._sawyer, self._end_effector)]
        ## position and velocity of prismatic joint
        joints = []
        self._state_query = bullet.format_sim_query(bodies, links, joints)

    def get_observation(self):
        observation = bullet.get_sim_state(*self._state_query)
        tip_distance = 1e3 * (bullet.get_joint_state(self._sawyer, 'right_gripper_l_finger_joint', 'pos') - bullet.get_joint_state(self._sawyer, 'right_gripper_r_finger_joint', 'pos'))
        return np.append(observation, np.array([[tip_distance]]))
