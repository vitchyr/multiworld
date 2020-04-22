from roboverse.envs.sawyer_grasp_v2 import SawyerGraspV2Env
import roboverse.bullet as bullet
import numpy as np
import gym

REWARD_NEGATIVE = 0.0
REWARD_POSITIVE = 1.0


class SawyerGraspV4Env(SawyerGraspV2Env):

    def _set_action_space(self):
        act_dim = 6
        # first three actions are delta x,y,z
        # action 4 is wrist rotation
        # action 5 is gripper open/close (> 0.5 for open, < -0.5 for close)
        # action 6 is terminate episode  (> 0.5 for termination)
        act_bound = 1
        act_high = np.ones(act_dim) * act_bound
        self.action_space = gym.spaces.Box(-act_high, act_high)

    def get_reward(self, info):
        object_list = self._objects.keys()
        reward = REWARD_NEGATIVE
        for object_name in object_list:
            object_info = bullet.get_body_info(self._objects[object_name],
                                               quat_to_deg=False)
            object_pos = np.asarray(object_info['pos'])
            object_height = object_pos[2]
            if object_height > self._reward_height_thresh:
                end_effector_pos = np.asarray(self.get_end_effector_pos())
                object_gripper_distance = np.linalg.norm(
                    object_pos - end_effector_pos)
                if object_gripper_distance < 0.1:
                    reward = REWARD_POSITIVE
        return reward

    def step(self, action):
        action = np.asarray(action)
        pos = list(bullet.get_link_state(self._sawyer, self._end_effector, 'pos'))
        delta_pos = action[:3]
        pos += delta_pos * self._action_scale
        pos = np.clip(pos, self._pos_low, self._pos_high)

        theta = list(bullet.get_link_state(self._sawyer, self._end_effector, 'theta'))
        delta_theta = action[3]
        target_theta = theta + np.asarray([0., 0., delta_theta*20])
        target_theta = np.clip(target_theta, [180, 0., 0.], [180, 0., 180.])
        target_theta = bullet.deg_to_quat(target_theta)

        gripper_action = action[4]

        # is_gripper_open = self._is_gripper_open()
        is_gripper_open = self._gripper_open
        if gripper_action > 0.5 and is_gripper_open:
            # keep it open
            gripper = -0.8
            self._simulate(pos, target_theta, gripper)
        elif gripper_action > 0.5 and not is_gripper_open:
            # gripper is currently closed and we want to open it
            gripper = -0.8
            for i in range(10):
                self._simulate(pos, target_theta, gripper)
            self._gripper_open = True
        elif gripper_action < -0.5 and not is_gripper_open:
            # keep it closed
            gripper = 0.8
            self._simulate(pos, target_theta, gripper)
        elif gripper_action < -0.5 and is_gripper_open:
            # gripper is open and we want to close it
            gripper = +0.8
            for i in range(10):
                self._simulate(pos, target_theta, gripper)
            self._gripper_open = False
        elif gripper_action <= 0.5 and gripper_action >= -0.5:
            # maintain current status
            if is_gripper_open:
                gripper = -0.8
            else:
                gripper = 0.8
            self._simulate(pos, target_theta, gripper)
            pass
        else:
            raise NotImplementedError

        # if self._visualize: self.visualize_targets(pos)

        if action[5] > 0.5:
            done = True
            reward = self.get_reward({})
            if reward > 0:
                info = {'grasp_success': 1.0}
            else:
                info = {'grasp_success': 0.0}
        else:
            done = False
            reward = REWARD_NEGATIVE
            info = {'grasp_success': 0.0}

        observation = self.get_observation()
        self._prev_pos = bullet.get_link_state(self._sawyer, self._end_effector,
                                               'pos')
        return observation, reward, done, info

    def reset(self):
        bullet.reset()
        bullet.setup_headless(self._timestep, solver_iterations=self._solver_iterations)
        self._load_meshes()
        self._format_state_query()

        self._prev_pos = np.array(self._pos_init)
        bullet.position_control(self._sawyer, self._end_effector, self._prev_pos, self.theta)

        # start with an open gripper
        self._gripper_open = True
        gripper = -0.8
        for _ in range(3):
            self._simulate(self._prev_pos, self.theta, gripper)

        return self.get_observation()


if __name__ == "__main__":
    import roboverse
    env = roboverse.make("SawyerGraspOneV4-v0",
                         gui=True,
                         observation_mode='state',)

    object_ind = 0
    for _ in range(50):
        obs = env.reset()
        # object_pos[2] = -0.30

        dist_thresh = 0.04 + np.random.normal(scale=0.01)

        for _ in range(25):
            ee_pos = obs[:3]
            object_pos = obs[object_ind * 7 + 8: object_ind * 7 + 8 + 3]
            # object_pos += np.random.normal(scale=0.02, size=(3,))

            object_gripper_dist = np.linalg.norm(object_pos - ee_pos)
            theta_action = 0.
            # theta_action = np.random.uniform()
            # print(object_gripper_dist)
            if object_gripper_dist > dist_thresh and env._gripper_open:
                # print('approaching')
                action = (object_pos - ee_pos) * 4.0
                action = np.concatenate((action, np.asarray([theta_action,0.,0.])))
            elif env._gripper_open:
                # print('gripper closing')
                action = (object_pos - ee_pos) * 4.0
                action = np.concatenate(
                    (action, np.asarray([0., -0.7, 0.])))
            elif object_pos[2] < env._reward_height_thresh:
                # print('lifting')
                action = np.asarray([0., 0., 0.7])
                action = np.concatenate(
                    (action, np.asarray([0., 0., 0.])))
            else:
                # print('terminating')
                action = np.asarray([0., 0., 0.7])
                action = np.concatenate(
                    (action, np.asarray([0., 0., 0.7])))

            action += np.random.normal(scale=0.1, size=(6,))
            # print(action)
            obs, rew, done, info = env.step(action)
            if rew > 0:
                print('reward: {}'.format(rew))
            if done:
                break

