import os
import time
import numpy as np
import gym
import pdb

import roboverse.bullet as bullet


class SawyerBaseEnv(gym.Env):

    def __init__(self,
                 img_dim=256,
                 gui=False,
                 action_scale=.1,
                 action_repeat=4,
                 timestep=1./120,
                 solver_iterations=150,
                 gripper_bounds=[-1,1],
                 visualize=False,
                 ):

        self._gui = gui
        self._action_scale = action_scale
        self._action_repeat = action_repeat
        self._timestep = timestep
        self._solver_iterations = solver_iterations
        self._gripper_bounds = gripper_bounds
        self._visualize = visualize

        bullet.connect_headless(self._gui)
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
        print('[ SawyerBase ] Resetting')
        bullet.reset()
        self._load_meshes()
        self._end_effector = bullet.get_index_by_attribute(
            self._sawyer, 'link_name', 'gripper_site')
        self._format_state_query()

        bullet.setup_headless(self._timestep, solver_iterations=self._solver_iterations)

        self._prev_pos = pos = np.array([0.5, 0, 0])
        self.theta = bullet.deg_to_quat([180, 0, 0])
        bullet.position_control(self._sawyer, self._end_effector, pos, self.theta)
        return self.get_observation()
    
    def open_gripper(self, act_repeat=10):
        delta_pos = [0,0,0]
        gripper = 0
        for _ in range(act_repeat):
            self.step(delta_pos, gripper)

    def _load_meshes(self):
        self._sawyer = bullet.objects.sawyer()
        self._table = bullet.objects.table()

    def _format_state_query(self):
        ## position and orientation of body root
        bodies = [self._cube, self._lid]
        ## position and orientation of link
        links = [(self._sawyer, self._end_effector)]
        ## position and velocity of prismatic joint
        joints = [(self._sawyer, None)]
        # joints = []
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

        if self._visualize: self.visualize_targets(pos)

        observation = self.get_observation()
        reward = self.get_reward(observation)
        done = self.get_termination(observation)
        self._prev_pos = pos
        return observation, reward, done, {}

    def _simulate(self, pos, theta, gripper):
        for _ in range(self._action_repeat):
            bullet.sawyer_ik(self._sawyer, self._end_effector, pos, self.theta, gripper, gripper_bounds=self._gripper_bounds, discrete_gripper=False)
            bullet.step_ik()

    def render(self, mode='rgb_array'):
        img, depth, segmentation = bullet.render(self._img_dim, self._img_dim, self._view_matrix, self._projection_matrix)
        return img

    def get_termination(self, observation):
        return False

    def get_reward(self, observation):
        return 0

    def visualize_targets(self, pos):
        bullet.add_debug_line(self._prev_pos, pos)

