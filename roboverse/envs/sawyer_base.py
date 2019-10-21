import os
import time
import numpy as np
import gym
import pdb

import roboverse.core as bullet


class SawyerBaseEnv(gym.Env):

    def __init__(self,
                 img_dim=256,
                 render=False,
                 action_scale=.1,
                 action_repeat=4,
                 timestep=1./120,
                 solver_iterations=150,
                 ):

        self._render = render
        self._action_scale = action_scale
        self._action_repeat = action_repeat
        self._timestep = timestep
        self._solver_iterations = solver_iterations

        bullet.connect_headless(self._render)
        self._set_spaces()
        
        self._img_dim = img_dim
        self._view_matrix = bullet.get_view_matrix()
        self._projection_matrix = bullet.get_projection_matrix(self._img_dim, self._img_dim)

    def _set_spaces(self):
        act_dim = 4
        act_bound = 1
        act_high = np.ones(act_dim) * act_bound
        self.action_space = gym.spaces.Box(-act_high, act_high)

        obs = self.reset()
        observation_dim = len(obs)
        obs_bound = 100
        obs_high = np.ones(observation_dim) * obs_bound
        self.observation_space = gym.spaces.Box(-obs_high, obs_high)

    def reset(self):
        bullet.reset()
        self._load_meshes()
        self._end_effector = bullet.get_index_by_attribute(
            self._sawyer, 'link_name', 'right_l6')
        self._format_state_query()

        bullet.setup_headless(self._timestep, solver_iterations=self._solver_iterations)

        pos = np.array([0.5, 0, 0])
        self.theta = [0.7071, 0.7071, 0, 0]
        bullet.position_control(self._sawyer, self._end_effector, pos, self.theta)
        return self.get_observation()

    def get_gripper_width(self):
        l_finger = bullet.get_index_by_attribute(self._sawyer, 'link_name', 'right_gripper_l_finger_tip')
        r_finger = bullet.get_index_by_attribute(self._sawyer, 'link_name', 'right_gripper_r_finger_tip')
        l_pos = bullet.get_link_state(self._sawyer, l_finger, 'pos')
        r_pos = bullet.get_link_state(self._sawyer, r_finger, 'pos')
        gripper_width = r_pos[1] - l_pos[1]
        return gripper_width 
    
    def open_gripper(self, act_repeat=10):
        delta_pos = [0,0,0]
        gripper = 0
        for _ in range(act_repeat):
            self.step(delta_pos, gripper)

    def _load_meshes(self):
        self._sawyer = bullet.objects.sawyer()
        self._table = bullet.objects.table()
        self._bowl = bullet.objects.bowl()
        self._cube = bullet.objects.spam()

    def _format_state_query(self):
        ## position and orientation of body root
        bodies = [self._cube]
        ## position and orientation of link
        links = [(self._sawyer, self._end_effector)]
        ## position and velocity of prismatic joint
        joints = [(self._sawyer, None), (self._bowl, 'lid_joint')]
        # joints = [(self._bowl, 'lid_joint')]
        self._state_query = bullet.format_sim_query(bodies, links, joints)

    def _format_action(self, *action):
        if len(action) == 1:
            delta_pos, gripper = action[0][:-1], action[0][-1]
        elif len(action) == 2:
            delta_pos, gripper = action[0], action[1]
        else:
            raise RuntimeError('Unrecognized action: {}'.format(action))
        return np.array(delta_pos), gripper

    def get_observation(self):
        observation = bullet.get_sim_state(*self._state_query)
        return observation

    def step(self, *action):
        delta_pos, gripper = self._format_action(*action)
        pos = bullet.get_link_state(self._sawyer, self._end_effector, 'pos')
        pos += delta_pos * self._action_scale

        self._simulate(pos, self.theta, gripper)

        observation = self.get_observation()
        reward = self.get_reward(observation)
        done = False
        return observation, reward, done, {}

    def _simulate(self, pos, theta, gripper):
        for _ in range(self._action_repeat):
            bullet.sawyer_ik(self._sawyer, self._end_effector, pos, self.theta, gripper, gripper_close_thresh=0)
            bullet.step_ik()

    def render(self, mode='rgb_array'):
        img, depth, segmentation = bullet.render(self._img_dim, self._img_dim, self._view_matrix, self._projection_matrix)
        return img

    def get_reward(self, observation):
        return 0


if __name__ == "__main__":
    env = SawyerBaseEnv(render=True)
    # env.reset()

    ## interactive
    import roboverse.devices as devices
    space_mouse = devices.SpaceMouse()
    space_mouse.start_control()

    while True:
        delta = space_mouse.control
        gripper = space_mouse.control_gripper
        # action = np.concatenate((delta, [gripper]))
        # print(action)
        obs = env.step(delta, gripper)

    ## drive toward cube
    # for i in range(500):
    #     cube_pos = np.array(bullet.get_body_info(env._cube, 'pos'))
    #     ee_pos = np.array(bullet.get_link_state(env._sawyer, env._end_effector, 'pos'))
    #     delta = cube_pos - ee_pos
    #     delta = np.clip(delta, -1, 1)
    #     print(delta)
    #     gripper = 0
    #     act = np.concatenate((delta, [gripper]))
    #     env.step(act)
    #     # pdb.set_trace()


    ## simple timing
    # num_steps = 100
    # t0 = time.time()
    # for i in range(num_steps):
    #     act = np.array([1.0, 0.0, 0.0, 0.0])
    #     obs = env.step(act)
    #     print(i, obs)
    # t1 = time.time()

    # tot_time = t1 - t0
    # fps = num_steps / tot_time
    # print('{} steps in {} seconds'.format(num_steps, tot_time))
    # print('{} fps'.format(fps))
