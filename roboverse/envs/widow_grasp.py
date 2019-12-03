import roboverse.bullet as bullet
import numpy as np
from roboverse.envs.widow_base import WidowBaseEnv


class WidowGraspOneEnv(WidowBaseEnv):

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

    def get_observation(self):
        left_tip_pos = bullet.get_link_state(
            self._robot_id, self._gripper_joint_name[0], keys='pos')
        right_tip_pos = bullet.get_link_state(
            self._robot_id, self._gripper_joint_name[1], keys='pos')
        left_tip_pos = np.asarray(left_tip_pos)
        right_tip_pos = np.asarray(right_tip_pos)

        gripper_tips_distance = [np.linalg.norm(
            left_tip_pos - right_tip_pos)]
        end_effector_pos = self.get_end_effector_pos()

        object_info = bullet.get_body_info(self._objects['lego'],
                                           quat_to_deg=False)
        object_pos = object_info['pos']
        object_theta = object_info['theta']

        return np.concatenate((end_effector_pos, gripper_tips_distance,
                               object_pos, object_theta))
