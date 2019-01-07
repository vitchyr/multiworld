import abc
import numpy as np
from gym.spaces import Box, Dict

from multiworld.core.serializable import Serializable
from multiworld.envs.mujoco.mujoco_env import MujocoEnv
from multiworld.core.multitask_env import MultitaskEnv
from multiworld.envs.env_util import get_asset_full_path

from collections import OrderedDict
from multiworld.envs.env_util import (
    get_stat_in_paths,
    create_stats_ordered_dict,
)


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
            *args,
            **kwargs):
        self.quick_init(locals())
        MultitaskEnv.__init__(self)
        MujocoEnv.__init__(self,
                           model_path=get_asset_full_path('classic_mujoco/normal_gear_ratio_ant.xml'),
                           frame_skip=frame_skip,
                           **kwargs)

        self.action_space = Box(-1 * np.ones(8),
                                1 * np.ones(8),
                                dtype=np.float32)
        self.ant_radius = None #0.30

        self.reward_type = reward_type
        self.norm_order = norm_order

        self.ant_low, self.ant_high = np.array(ant_low), np.array(ant_high)
        self.goal_low, self.goal_high = np.array(goal_low), np.array(goal_high)
        self.two_frames = two_frames
        self.vel_in_state = vel_in_state
        if self.vel_in_state:
            obs_space_low = np.concatenate((self.ant_low, -1 * np.ones(27)))
            obs_space_high = np.concatenate((self.ant_high, 1 * np.ones(27)))
            goal_space_low = np.concatenate((self.goal_low, -1 * np.ones(27)))
            goal_space_high = np.concatenate((self.goal_high, 1 * np.ones(27)))
        else:
            obs_space_low = np.concatenate((self.ant_low, -1 * np.ones(13)))
            obs_space_high = np.concatenate((self.ant_high, 1 * np.ones(13)))
            goal_space_low = np.concatenate((self.goal_low, -1 * np.ones(13)))
            goal_space_high = np.concatenate((self.goal_high, 1 * np.ones(13)))

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

        self._state_goal = None
        self._prev_obs = None
        self.reset()

    def step(self, action):
        self._prev_obs = self._cur_obs
        self.do_simulation(np.array(action), self.frame_skip)
        ob = self._get_obs()
        reward = self.compute_reward(action, ob)
        state, goal = ob['state_observation'], ob['state_desired_goal']
        full_state_diff = np.linalg.norm(state - goal)
        info = {
            'full_state_diff': full_state_diff,
        }
        # if self.vel_in_state:
        #     info['velocity_diff'] = np.linalg.norm(state[-4:-1] - goal[-4:-1])
        #     info['angular_velocity_diff'] = np.linalg.norm(state[-1] - goal[-1])
        done = False
        return ob, reward, done, info

    def _get_obs(self):
        qpos = list(self.sim.data.qpos.flat)
        flat_obs = qpos
        if self.vel_in_state:
            flat_obs = flat_obs + list(self.sim.data.qvel.flat)
        flat_obs = np.array(flat_obs)

        self._cur_obs = dict(
            observation=flat_obs,
            desired_goal=self._state_goal,
            achieved_goal=flat_obs,
            state_observation=flat_obs,
            state_desired_goal=self._state_goal,
            state_achieved_goal=flat_obs,
            proprio_observation=flat_obs,
            proprio_desired_goal=self._state_goal,
            proprio_achieved_goal=flat_obs,
        )

        if self.two_frames:
            if self._prev_obs is None:
                self._prev_obs = self._cur_obs
            frames = self.merge_frames(self._prev_obs, self._cur_obs)
            return frames

        return self._cur_obs

    def merge_frames(self, dict1, dict2):
        dict = {}
        for key in dict1.keys():
            dict[key] = np.concatenate((dict1[key], dict2[key]))
        return dict

    def get_goal(self):
        if self.two_frames:
            return {
                'desired_goal': np.concatenate((self._state_goal, self._state_goal)),
                'state_desired_goal': np.concatenate((self._state_goal, self._state_goal)),
            }
        else:
            return {
                'desired_goal': self._state_goal,
                'state_desired_goal': self._state_goal,
            }

    def sample_goals(self, batch_size):
        goals = np.random.uniform(
            self.goal_space.low,
            self.goal_space.high,
            size=(batch_size, self.goal_space.low.size),
        )
        if self.two_frames:
            goals = goals[:,:int(self.goal_space.low.size/2)]

        if self.two_frames:
            return {
                'desired_goal': np.concatenate((goals, goals), axis=1),
                'state_desired_goal': np.concatenate((goals, goals), axis=1),
            }
        else:
            return {
                'desired_goal': goals,
                'state_desired_goal': goals,
            }

    def compute_rewards(self, actions, obs):
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
        self.set_goal(self.sample_goal())
        self.sim.forward()
        self._prev_obs = None
        self._cur_obs = None
        return self._get_obs()

    def _reset_ant(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        pos_low = np.array([-2, -2, 0.25])
        pos_high = np.array([2, 2, 1.75])
        qpos[:3] = np.random.uniform(pos_low, pos_high)

        quat_low = np.array([-1, -1, -1, -1])
        quat_high = np.array([1, 1, 1, 1])
        qpos[3:7] = np.random.uniform(quat_low, quat_high)

        hip_and_leg_low = np.deg2rad([-30, 30, -30, -70, -30, -70, -30, 30])
        hip_and_leg_high = np.deg2rad([30, 70, 30, -30, 30, -30, 30, 70])
        qpos[-8:] = np.random.uniform(hip_and_leg_low, hip_and_leg_high)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)

    def set_goal(self, goal):
        if self.two_frames:
            self._state_goal = goal['state_desired_goal'][int(len(goal['state_desired_goal'])/2):]
        else:
            self._state_goal = goal['state_desired_goal']
        self._prev_obs = None
        self._cur_obs = None

    def set_to_goal(self, goal):
        state_goal = goal['state_desired_goal']
        if self.two_frames:
            state_goal = state_goal[:int(len(state_goal)/2)]
        qpos, qvel = np.zeros(15), np.zeros(14)
        qpos = state_goal[:15]
        if self.vel_in_state:
            qvel = state_goal[15:]
        self.set_state(qpos, qvel)
        self._prev_obs = None
        self._cur_obs = None

    def get_env_state(self):
        joint_state = self.sim.get_state()
        goal = self._state_goal.copy()
        return joint_state, goal, self._prev_obs

    def set_env_state(self, state):
        state, goal, prev_obs = state
        self.sim.set_state(state)
        self.sim.forward()
        self._state_goal = goal
        self._prev_obs = prev_obs

    def valid_state(self, state):
        pass
        # return self.valid_states(state[None])[0]

    def valid_states(self, states):
        pass
        # states[:,3] = np.clip(states[:,3], -1, 1) #sin
        # states[:,4] = np.clip(states[:,4], -1, 1) #cos
        # angle = np.arcsin(states[:,3])
        # for i in range(len(angle)):
        #     if states[i][4] <= 0:
        #         angle[i] = np.pi - angle[i]
        # states[:,3], states[:,4] = np.sin(angle), np.cos(angle)
        # return states

    def get_diagnostics(self, paths, prefix=''):
        statistics = OrderedDict()
        list_of_stat_names = [
            'full_state_diff',
            # 'pos_diff',
            # 'angle_diff',
            # 'pos_angle_diff',
        ]
        # if self.vel_in_state:
        #     list_of_stat_names.append('velocity_diff')
        #     list_of_stat_names.append('angular_velocity_diff')

        for stat_name in list_of_stat_names:
            stat_name = stat_name
            stat = get_stat_in_paths(paths, 'env_infos', stat_name)
            statistics.update(create_stats_ordered_dict(
                '%s%s' % (prefix, stat_name),
                stat,
                always_show_all_stats=True,
                ))
            statistics.update(create_stats_ordered_dict(
                'Final %s%s' % (prefix, stat_name),
                [s[-1] for s in stat],
                always_show_all_stats=True,
                ))
        return statistics

    def viewer_setup(self):
        # self.viewer.cam.trackbodyid = 0
        # self.viewer.cam.lookat[0] = 0.0
        # self.viewer.cam.lookat[1] = 0.0
        # self.viewer.cam.lookat[2] = 0.5
        # self.viewer.cam.distance = 6.5
        # self.viewer.cam.elevation = -90
        self.viewer.cam.distance = self.model.stat.extent * 0.5