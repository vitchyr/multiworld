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
            goal_is_qpos=False,
            init_qpos=None,
            fixed_goal=None,
            init_xy_mode='corner',
            terminate_when_unhealthy=False,
            healthy_z_range=(0.2, 1.0),
            *args,
            **kwargs):
        assert init_xy_mode in {
            'corner',
            'sample-uniformly-xy-space',
            'sample-from-goal-space',  # soon to be deprecated
        }
        assert not goal_is_xy or not goal_is_qpos
        self.quick_init(locals())
        MultitaskEnv.__init__(self)
        MujocoEnv.__init__(self,
                           model_path=get_asset_full_path(model_path),
                           frame_skip=frame_skip,
                           **kwargs)
        if goal_is_xy:
            assert reward_type.startswith('xy')

        if init_qpos is not None:
            self.init_qpos[:len(init_qpos)] = np.array(init_qpos)

        self.action_space = Box(-np.ones(8), np.ones(8), dtype=np.float32)
        self.reward_type = reward_type
        self.norm_order = norm_order
        self.goal_is_xy = goal_is_xy
        self.goal_is_qpos = goal_is_qpos
        self.fixed_goal = fixed_goal
        self.init_xy_mode = init_xy_mode
        self.terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range

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
        qpos_space = Box(-10*np.ones(15), 10*np.ones(15))

        spaces = [
            ('observation', self.obs_space),
            ('desired_goal', self.goal_space),
            ('achieved_goal', self.goal_space),
            ('state_observation', self.obs_space),
            ('state_desired_goal', self.goal_space),
            ('state_achieved_goal', self.goal_space),
        ]
        if self.goal_is_xy:
            spaces += [
                ('xy_observation', self.obs_space),
                ('xy_desired_goal', self.goal_space),
                ('xy_achieved_goal', self.goal_space),
            ]
        if self.goal_is_qpos:
            spaces += [
                ('qpos_desired_goal', qpos_space),
                ('qpos_achieved_goal', qpos_space),
            ]

        self.observation_space = Dict(spaces)

        self._full_state_goal = None
        self._xy_goal = None
        self._qpos_goal = None
        self._prev_obs = None
        self._cur_obs = None

    def step(self, action):
        self._prev_obs = self._cur_obs
        self.do_simulation(np.array(action), self.frame_skip)
        ob = self._get_obs()
        reward = self.compute_reward(action, ob)
        info = {
            'xy-distance': self._compute_xy_distances(
                self.numpy_batchify_dict(ob)
            ),
            'full-state-distance': self._compute_state_distances(
                self.numpy_batchify_dict(ob)
            ),
            'qpos-distance': self._compute_qpos_distances(
                self.numpy_batchify_dict(ob)
            ),
        }
        if self.terminate_when_unhealthy:
            done = not self.is_healthy
        else:
            done = False
        self._cur_obs = ob
        if len(self.init_qpos) > 15 and self.viewer is not None:
            qpos = self.sim.data.qpos
            qpos[15:] = self._full_state_goal[:15]
            qvel = self.sim.data.qvel
            self.set_state(qpos, qvel)
        return ob, reward, done, info

    @property
    def is_healthy(self):
        state = self.state_vector()
        min_z, max_z = self._healthy_z_range
        is_healthy = (np.isfinite(state).all() and min_z <= state[2] <= max_z)
        return is_healthy

    def _get_obs(self):
        qpos = list(self.sim.data.qpos.flat)[:15]
        flat_obs = qpos
        if self.vel_in_state:
            flat_obs = flat_obs + list(self.sim.data.qvel.flat[:14])

        xy = self.sim.data.get_body_xpos('torso')[:2]
        ob = dict(
            observation=flat_obs,
            desired_goal=self._full_state_goal,
            achieved_goal=flat_obs,
            state_observation=flat_obs,
            state_desired_goal=self._full_state_goal,
            state_achieved_goal=flat_obs,
            xy_observation=xy,
            xy_desired_goal=self._xy_goal,
            xy_achieved_goal=xy,
            qpos_desired_goal=self._qpos_goal,
            qpos_achieved_goal=qpos,
        )

        if self.two_frames:
            if self._prev_obs is None:
                self._prev_obs = ob
            ob = self.merge_frames(self._prev_obs, ob)

        # Make sure a copy of the observation is used to avoid aliasing bugs.
        ob = {k: np.array(v) for k, v in ob.items()}
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
        goals = self._sample_uniform_xy(batch_size)
        if self.two_frames:
            goals = goals[:, :int(self.goal_space.low.size/2)]
        return goals

    def _sample_uniform_xy(self, batch_size):
        goals = np.random.uniform(
            self.goal_space.low[:2],
            self.goal_space.high[:2],
            size=(batch_size, 2),
        )
        return goals

    def compute_rewards(self, actions, obs):
        if self.reward_type == 'xy_dense':
            r = - self._compute_xy_distances(obs)
        elif self.reward_type == 'dense':
            r = - self._compute_state_distances(obs)
        elif self.reward_type == 'qpos_dense':
            r = - self._compute_qpos_distances(obs)
        elif self.reward_type == 'vectorized_dense':
            r = - self._compute_vectorized_state_distances(obs)
        else:
            raise NotImplementedError("Invalid/no reward type.")
        return r

    def _compute_xy_distances(self, obs):
        achieved_goals = obs['xy_achieved_goal']
        desired_goals = obs['xy_desired_goal']
        diff = achieved_goals - desired_goals
        return np.linalg.norm(diff, ord=self.norm_order, axis=1)

    def _compute_state_distances(self, obs):
        achieved_goals = obs['state_achieved_goal']
        desired_goals = obs['state_desired_goal']
        if desired_goals.shape == (1,):
            return -1000
        ant_pos = achieved_goals
        goals = desired_goals
        # import ipdb; ipdb.set_trace()
        diff = ant_pos - goals
        return np.linalg.norm(diff, ord=self.norm_order, axis=1)

    def _compute_qpos_distances(self, obs):
        achieved_goals = obs['qpos_achieved_goal']
        desired_goals = obs['qpos_desired_goal']
        if desired_goals.shape == (1,):
            return -1000
        return np.linalg.norm(
            achieved_goals - desired_goals, ord=self.norm_order, axis=1
        )

    def _compute_vectorized_state_distances(self, obs):
        achieved_goals = obs['state_achieved_goal']
        desired_goals = obs['state_desired_goal']
        ant_pos = achieved_goals
        goals = desired_goals
        diff = ant_pos - goals
        return np.abs(diff)

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
        if self.init_xy_mode == 'sample-uniformly-xy-space':
            xy_start = self._sample_uniform_xy(1)[0]
            qpos[:2] = xy_start
        self.set_state(qpos, qvel)

    def _set_goal(self, goal):
        if self.goal_is_xy:
            self._xy_goal = goal['xy_desired_goal']
        else:
            self._xy_goal = goal['state_desired_goal'][:2]
        if not self.goal_is_xy:
            if self.two_frames:
                self._full_state_goal = goal['state_desired_goal'][int(len(goal['state_desired_goal']) / 2):]
            else:
                self._full_state_goal = goal['state_desired_goal']
            if self.goal_is_qpos:
                self._qpos_goal = self._full_state_goal[:15]
        self._prev_obs = None
        self._cur_obs = None
        if len(self.init_qpos) > 15:
            qpos = self.init_qpos
            qpos[15:] = self._full_state_goal[:15]
            qvel = self.sim.data.qvel
            self.set_state(qpos, qvel)
        else:
            site_xpos = self.sim.data.site_xpos
            goal_xpos = np.concatenate((self._xy_goal, np.array([0.5])))
            site_xpos[self.sim.model.site_name2id('goal')] = goal_xpos
            self.model.site_pos[:] = site_xpos


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
