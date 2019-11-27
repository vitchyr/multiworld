import roboverse.bullet as bullet
import numpy as np
from roboverse.envs.sawyer_base import SawyerBaseEnv


class SawyerGraspOneEnv(SawyerBaseEnv):

    def __init__(self,
                 goal_pos=(0.75, 0.2, -0.1),
                 reward_type='shaped',
                 *args,
                 **kwargs
                 ):

        super().__init__(*args, **kwargs)
        self._goal_pos = goal_pos
        self._reward_type = reward_type
        self.dt = 0.1

    def _load_meshes(self):
        super()._load_meshes()
        self._objects = {
            'lego': bullet.objects.lego()
        }

    def step(self, *action):
        delta_pos, gripper = self._format_action(*action)
        pos = bullet.get_link_state(self._robot_id, self._end_effector, 'pos')
        pos += delta_pos * self._action_scale
        pos = np.clip(pos, self._pos_low, self._pos_high)

        self._simulate(pos, self.theta, gripper)
        if self._visualize: self.visualize_targets(pos)

        observation = self.get_observation()
        info = self.get_info()
        reward = self.get_reward(info)
        done = self.get_termination(observation)
        self._prev_pos = bullet.get_link_state(self._robot_id, self._end_effector, 'pos')
        return observation, reward, done, info

    def get_info(self):
        object_pos = self.get_object_midpoint('lego')
        object_goal_distance = np.linalg.norm(object_pos - self._goal_pos)
        end_effector_pos = self.get_end_effector_pos()
        object_gripper_distance = np.linalg.norm(
            object_pos - end_effector_pos)
        info = {
            'object_goal_distance': object_goal_distance,
            'object_gripper_distance': object_gripper_distance
        }
        return info

    def get_reward(self, info):

        if self._reward_type == 'sparse':
            if info['object_goal_distance'] < 0.1:
                reward = 1
            else:
                reward = 0
        elif self._reward_type == 'shaped':
            reward = -1*(4*info['object_goal_distance']
                         + info['object_gripper_distance'])
        else:
            raise NotImplementedError

        return reward

    def get_observation(self):
        left_tip_pos = bullet.get_link_state(
            self._robot_id, 'right_gripper_l_finger_joint', keys='pos')
        right_tip_pos = bullet.get_link_state(
            self._robot_id, 'right_gripper_r_finger_joint', keys='pos')
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
