import abc
import numpy as np
from gym.spaces import Box, Dict

from multiworld.core.serializable import Serializable
from multiworld.envs.mujoco.mujoco_env import MujocoEnv
from multiworld.core.multitask_env import MultitaskEnv
from multiworld.envs.env_util import get_asset_full_path


class AntEnv(MujocoEnv, Serializable, MultitaskEnv, metaclass=abc.ABCMeta):
    def __init__(
            self,
            reward_type='dense',
            norm_order=2,
            frame_skip=5,
            two_frames=False,
            vel_in_state=True,
            ant_low=list([-1.60, -1.60]),
            ant_high=list([1.60, 1.60]),
            goal_low=list([-1.60, -1.60]),
            goal_high=list([1.60, 1.60]),
            model_path='classic_mujoco/normal_gear_ratio_ant.xml',
            goal_is_xy=False,
            init_qpos=None,
            fixed_goal=None,
            *args,
            **kwargs):
        self.quick_init(locals())
        MultitaskEnv.__init__(self)
        MujocoEnv.__init__(self,
                           model_path=get_asset_full_path(model_path),
                           frame_skip=frame_skip,
                           **kwargs)
        if goal_is_xy:
            assert reward_type.startswith('xy')

        if init_qpos is not None:
            self.init_qpos = np.array(init_qpos)

        self.action_space = Box(-np.ones(8), np.ones(8), dtype=np.float32)
        self.reward_type = reward_type
        self.norm_order = norm_order
        self.goal_is_xy = goal_is_xy
        self.fixed_goal = fixed_goal

        self.ant_low, self.ant_high = np.array(ant_low), np.array(ant_high)
        goal_low, goal_high = np.array(goal_low), np.array(goal_high)
        self.two_frames = two_frames
        self.vel_in_state = vel_in_state
        if self.vel_in_state:
            obs_space_low = np.concatenate((self.ant_low, -np.ones(27)))
            obs_space_high = np.concatenate((self.ant_high, np.ones(27)))
            if goal_is_xy:
                goal_space_low = goal_low
                goal_space_high = goal_high
            else:
                goal_space_low = np.concatenate((goal_low, -np.ones(27)))
                goal_space_high = np.concatenate((goal_high, np.ones(27)))
        else:
            obs_space_low = np.concatenate((self.ant_low, -np.ones(13)))
            obs_space_high = np.concatenate((self.ant_high, np.ones(13)))
            if goal_is_xy:
                goal_space_low = goal_low
                goal_space_high = goal_high
            else:
                goal_space_low = np.concatenate((goal_low, -np.ones(13)))
                goal_space_high = np.concatenate((goal_high, np.ones(13)))

        if self.two_frames:
            self.obs_space = Box(np.concatenate((obs_space_low, obs_space_low)),
                                 np.concatenate((obs_space_high, obs_space_high)),
                                 dtype=np.float32)
            self.goal_space = Box(np.concatenate((goal_space_low, goal_space_low)),
                                  np.concatenate((goal_space_high, goal_space_high)),
                                  dtype=np.float32)
        else:
            self.obs_space = Box(obs_space_low, obs_space_high, dtype=np.float32)
            self.goal_space = Box(goal_space_low, goal_space_high, dtype=np.float32)

        self.observation_space = Dict([
            ('observation', self.obs_space),
            ('desired_goal', self.goal_space),
            ('achieved_goal', self.obs_space),
            ('state_observation', self.obs_space),
            ('state_desired_goal', self.goal_space),
            ('state_achieved_goal', self.obs_space),
            ('proprio_observation', self.obs_space),
            ('proprio_desired_goal', self.goal_space),
            ('proprio_achieved_goal', self.obs_space),
        ])

        self._full_state_goal = None
        self._xy_goal = None
        self._prev_obs = None
        self._cur_obs = None
        self.reset()

    def step(self, action):
        self._prev_obs = self._cur_obs
        self.do_simulation(np.array(action), self.frame_skip)
        ob = self._get_obs()
        reward = self.compute_reward(action, ob)
        # state, goal = ob['state_observation'], ob['state_desired_goal']
        # full_state_diff = np.linalg.norm(state - goal)
        # info = {
        #     'full_state_diff': full_state_diff,
        # }
        # if self.vel_in_state:
        #     info['velocity_diff'] = np.linalg.norm(state[-4:-1] - goal[-4:-1])
        #     info['angular_velocity_diff'] = np.linalg.norm(state[-1] - goal[-1])
        info = {}
        done = False
        self._cur_obs = ob
        return ob, reward, done, info

    def _get_obs(self):
        qpos = list(self.sim.data.qpos.flat)
        flat_obs = qpos
        if self.vel_in_state:
            flat_obs = flat_obs + list(self.sim.data.qvel.flat)
        flat_obs = np.array(flat_obs)

        xy = self.sim.data.get_body_xpos('torso')[:2]
        ob = dict(
            observation=flat_obs,
            desired_goal=self._full_state_goal,
            achieved_goal=flat_obs,
            state_observation=flat_obs,
            state_desired_goal=self._full_state_goal,
            state_achieved_goal=flat_obs,
            proprio_observation=flat_obs,
            proprio_desired_goal=self._full_state_goal,
            proprio_achieved_goal=flat_obs,
            xy_observation=xy,
            xy_desired_goal=self._xy_goal,
            xy_achieved_goal=xy,
        )

        if self.two_frames:
            if self._prev_obs is None:
                self._prev_obs = ob
            frames = self.merge_frames(self._prev_obs, ob)
            return frames

        return ob

    def merge_frames(self, dict1, dict2):
        dict = {}
        for key in dict1.keys():
            dict[key] = np.concatenate((dict1[key], dict2[key]))
        return dict

    def get_goal(self):
        if self.two_frames:
            return {
                'desired_goal': np.concatenate((self._full_state_goal, self._full_state_goal)),
                'state_desired_goal': np.concatenate((self._full_state_goal, self._full_state_goal)),
                'xy_desired_goal': np.concatenate((self._xy_goal, self._xy_goal)),
            }
        else:
            return {
                'desired_goal': self._full_state_goal,
                'state_desired_goal': self._full_state_goal,
                'xy_desired_goal': self._xy_goal,
            }

    def sample_goals(self, batch_size):
        if self.fixed_goal is not None:
            goals = np.array(self.fixed_goal)[None].repeat(batch_size, axis=0)
        else:
            goals = self._sample_random_goal_vectors(batch_size)
        if self.goal_is_xy:
            goals_dict = {
                'xy_desired_goal': goals,
            }
        else:
            if self.two_frames:
                goals_dict = {
                    'desired_goal': np.concatenate((goals, goals), axis=1),
                    'state_desired_goal': np.concatenate((goals, goals), axis=1),
                }
            else:
                goals_dict = {
                    'desired_goal': goals,
                    'state_desired_goal': goals,
                }

        return goals_dict

    def _sample_random_goal_vectors(self, batch_size):
        goals = np.random.uniform(
            self.goal_space.low,
            self.goal_space.high,
            size=(batch_size, self.goal_space.low.size),
        )
        if self.two_frames:
            goals = goals[:,:int(self.goal_space.low.size/2)]

        print(self.goal_space.low, self.goal_space.high)
        return goals

    def compute_rewards(self, actions, obs):
        if self.reward_type == 'xy_dense':
            achieved_goals = obs['xy_achieved_goal']
            desired_goals = obs['xy_desired_goal']
            diff = achieved_goals - desired_goals
            r = -np.linalg.norm(diff, ord=self.norm_order, axis=1)
        else:
            achieved_goals = obs['state_achieved_goal']
            desired_goals = obs['state_desired_goal']
            ant_pos = achieved_goals
            goals = desired_goals
            diff = ant_pos - goals
            if self.reward_type == 'dense':
                r = -np.linalg.norm(diff, ord=self.norm_order, axis=1)
            elif self.reward_type == 'vectorized_dense':
                r = -np.abs(diff)
            else:
                raise NotImplementedError("Invalid/no reward type.")
        return r

    def reset_model(self):
        self._reset_ant()
        self._set_goal(self.sample_goal())
        self.sim.forward()
        self._prev_obs = None
        self._cur_obs = None
        return self._get_obs()

    def _reset_ant(self):
        qpos = self.init_qpos
        qvel = np.zeros_like(self.init_qvel)
        self.set_state(qpos, qvel)

    def _set_goal(self, goal):
        if self.goal_is_xy:
            self._xy_goal = goal['xy_desired_goal']
            site_xpos = self.sim.data.site_xpos
            goal_xpos = np.concatenate((self._xy_goal, np.array([0.5])))
            site_xpos[self.sim.model.site_name2id('goal')] = goal_xpos
            self.model.site_pos[:] = site_xpos

        else:
            if self.two_frames:
                self._full_state_goal = goal['state_desired_goal'][int(len(goal['state_desired_goal']) / 2):]
            else:
                self._full_state_goal = goal['state_desired_goal']
        self._prev_obs = None
        self._cur_obs = None


if __name__ == '__main__':
    env = AntEnv(
        model_path='classic_mujoco/ant_maze.xml',
        goal_low=[-1, -1],
        goal_high=[6, 6],
        goal_is_xy=True,
        init_qpos=[
            0, 0, 0.5, 1,
            0, 0, 0,
            0,
            1.,
            0.,
            -1.,
            0.,
            -1.,
            0.,
            1.,
        ],
        reward_type='xy_dense',
    )
    env.reset()
    i = 0
    while True:
        i += 1
        env.render()
        action = env.action_space.sample()
        # action = np.zeros_like(action)
        env.step(action)
        if i % 10 == 0:
            env.reset()
