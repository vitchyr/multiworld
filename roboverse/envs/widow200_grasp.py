import roboverse.bullet as bullet
import numpy as np
from roboverse.envs.widow_base import WidowBaseEnv
from roboverse.utils.shapenet_utils import load_single_object


class Widow200GraspEnv(WidowBaseEnv):
    def __init__(self, goal_pos=(.7, 0.15, -0.20), *args, **kwargs):
        self._env_name = 'WidowX200GraspEnv'
        kwargs['downwards'] = False
        super().__init__(*args, **kwargs)
        self._goal_pos = goal_pos
        self._reward_type = 'shaped'
        self.RESET_JOINTS = [1.57, -0.6, -0.6, -1.57, 1.57]
        self._end_effector = 8

    def _load_meshes(self):
        super()._load_meshes()
        if self._env_name == "WidowX200GraspEnv":
            self._objects = {
                'lego': bullet.objects.lego(),
                'box': bullet.objects.box(),
            }

    def _simulate(self, pos, theta, gripper, delta_theta,
                  discrete_gripper=True):
        wrist_theta = delta_theta
        for _ in range(self._action_repeat):
            bullet.sawyer_position_theta_ik(
                self._robot_id, self._end_effector, pos, theta, gripper,
                wrist_theta, gripper_name=self._gripper_joint_name,
                gripper_bounds=self._gripper_bounds,
                discrete_gripper=discrete_gripper, max_force=self._max_force
            )
            bullet.step_ik(self._gripper_range)

    def reset(self):
        bullet.reset()
        self._load_meshes()
        # Allow the objects to settle down after they are dropped in sim
        for _ in range(50):
            bullet.step()

        self._format_state_query()

        bullet.setup_headless(self._timestep,
                              solver_iterations=self._solver_iterations)
        for i in range(len(self.RESET_JOINTS)):
            bullet.p.resetJointState(self._robot_id, i, self.RESET_JOINTS[i])
        self._prev_pos, self.theta = bullet.p.getLinkState(
            self._robot_id, 5, computeForwardKinematics=1)[4:6]

        for _ in range(5):
            pos = list(bullet.get_link_state(self._robot_id, self._end_effector,
                                             'pos'))
            theta = list(
                bullet.get_link_state(self._robot_id, self._end_effector,
                                      'theta'))
            gripper = -0.8
            self._simulate(pos, theta, gripper, delta_theta=0.)

        return self.get_observation()

    def get_reward(self, info):
        if self._reward_type == 'sparse':
            if info['object_goal_distance'] < 0.1:
                reward = 1
            else:
                reward = 0
        elif self._reward_type == 'shaped':
            reward = -1 * (4 * info['object_goal_distance']
                           + info['object_gripper_distance'])
        else:
            raise NotImplementedError

        return reward

    def step(self, *action):
        delta_pos, gripper = self._format_action(*action)
        pos = bullet.get_link_state(self._robot_id, self._end_effector, 'pos')

        # Debug
        q_indices = [bullet.get_joint_info(self._robot_id, j, 'joint_name')
                     for j in range(10)]
        # print("q_indices", q_indices)
        pos += delta_pos * self._action_scale
        pos = np.clip(pos, self._pos_low, self._pos_high)

        self._simulate(pos, self.theta, gripper)
        if self._visualize: self.visualize_targets(pos)

        observation = self.get_observation()
        info = self.get_info()
        reward = self.get_reward(info)
        done = self.get_termination(observation)
        self._prev_pos = bullet.get_link_state(self._robot_id,
                                               self._end_effector, 'pos')
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

        if 'lego' not in self._objects:
            return np.concatenate((end_effector_pos, gripper_tips_distance))

        object_info = bullet.get_body_info(self._objects['lego'],
                                           quat_to_deg=False)
        object_pos = object_info['pos']
        object_theta = object_info['theta']

        return np.concatenate((end_effector_pos, gripper_tips_distance,
                               object_pos, object_theta))
