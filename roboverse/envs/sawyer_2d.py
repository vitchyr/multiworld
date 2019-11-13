import copy
import numpy as np
import gym
import pdb

import roboverse as rv

class Sawyer2dEnv(gym.Env):

    def __init__(self, env='SawyerSoup-v0',
                       objects=['cube', 'lid'],
                       pos_init=[.75, 0, 0],
                       pos_high=[.75,.4,.25],
                       pos_low=[.75,-.6,-.36],
                       **kwargs):

        self._env = gym.make(env, pos_init=pos_init, pos_high=pos_high, pos_low=pos_low, **kwargs)
        self._objects = ['_' + obj for obj in objects]
        self._init_states = self._get_body_states(self._objects)
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        self._format_state_query()
        # self._target_x = {obj: pos[0] for obj, pos in target_states.items()}
        # pdb.set_trace()
        # self._set_spaces()

    # def __getattr__(self, attr):
    #     return getattr(self._env, attr)

    # def _set_spaces():
    #     act_low = self._env.action_space.low[1:]
    #     act_high = self._env.action_space.high[1:]
    #     self.action_space = gym.spaces.Box(-act_high, act_high)

    def _format_state_query(self):
        ## position and orientation of body root
        # bodies = [getattr(self._env, obj) for obj in self._objects]
        bodies = []
        ## position and orientation of link
        links = [(self._env._sawyer, self._env._end_effector)]
        ## position and velocity of prismatic joint
        joints = []
        self._env._state_query = rv.bullet.format_sim_query(bodies, links, joints)

    def reset(self, *args, **kwargs):
        return self._env.reset(*args, **kwargs)

    def step(self, act, *args, **kwargs):
        act[0] = 0
        out = self._env.step(act, *args, **kwargs)

        current_states = self._get_body_states(self._objects)
        for obj in self._objects:
            current = current_states[obj]
            init = self._init_states[obj]

            x = init['pos'][0:1]
            yz = current['pos'][1:3]
            theta = current['theta']
            pos = x + yz
            body = getattr(self._env, obj)
            rv.bullet.set_body_state(body, pos, theta)

        obs = self._env.get_observation()
        rew = self._env.get_reward(obs)
        term = self._env.get_termination(obs)
        info = {}
        return obs, rew, term, info

    def _get_body_states(self, objects):
        states = {}
        for obj in objects:
            body = getattr(self._env, obj)
            state = rv.bullet.get_body_info(body, ['pos', 'theta'])
            states[obj] = state
        return states

    # def __init__(self, goal_pos=[.75,-.4,.2], *args, goal_mult=4, bonus=0, min_reward=-3., **kwargs):
    #     self.record_args(locals())
    #     super().__init__(*args, **kwargs)
    #     self._goal_pos = goal_pos
    #     self._goal_mult = goal_mult
    #     self._bonus = bonus
    #     self._min_reward = min_reward
    #     self._id = 'SawyerLiftEnv'

    # def get_params(self):
    #     params = super().get_params()
    #     labels = ['_goal_pos', '_goal_mult', '_min_reward']
    #     params.update({label: getattr(self, label) for label in labels})
    #     return params

    # def _load_meshes(self):
    #     super()._load_meshes()
    #     self._bowl = bullet.objects.bowl()
    #     self._lid = bullet.objects.lid()
    #     self._cube = bullet.objects.spam()

    # def get_reward(self, observation):
    #     cube_pos = bullet.get_midpoint(self._cube)
    #     ee_pos = bullet.get_link_state(self._sawyer, self._end_effector, 'pos')
    #     ee_dist = bullet.l2_dist(cube_pos, ee_pos)
    #     goal_dist = bullet.l2_dist(cube_pos, self._goal_pos)
    #     reward = -(ee_dist + self._goal_mult * goal_dist)
    #     reward = max(reward, self._min_reward)
    #     if goal_dist < 0.25:
    #         reward += self._bonus
    #     # print(self._sensor_lid.sense(), self._sensor_cube.sense())
    #     return reward

