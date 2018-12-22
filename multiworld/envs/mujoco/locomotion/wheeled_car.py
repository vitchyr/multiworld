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


class WheeledCarEnv(MujocoEnv, Serializable, MultitaskEnv, metaclass=abc.ABCMeta):
    def __init__(
            self,
            reward_type='dense',
            norm_order=2,
            action_scale=20,
            frame_skip=3,
            car_low=list([-1.60, -1.60]),
            car_high=list([1.60, 1.60]),
            goal_low=list([-1.60, -1.60]),
            goal_high=list([1.60, 1.60]),
            *args,
            **kwargs):
        self.quick_init(locals())
        MultitaskEnv.__init__(self)
        MujocoEnv.__init__(self,
                           model_path=get_asset_full_path('locomotion/wheeled_car.xml'),
                           frame_skip=frame_skip,
                           **kwargs)

        self.action_space = Box(np.array([-1, -1]), np.array([1, 1]), dtype=np.float32)
        self.action_scale = action_scale
        self.ball_radius = 0.30


        self.reward_type = reward_type
        self.norm_order = norm_order

        self.car_low, self.car_high = np.array(car_low), np.array(car_high)
        self.goal_low, self.goal_high = np.array(goal_low), np.array(goal_high)
        self.obs_space = Box(np.concatenate((self.car_low, [-1, -1, -1, -10, -10, -10, -10])),
                             np.concatenate((self.car_high, [0.03, 1, 1, 10, 10, 10, 10])),
                             dtype=np.float32)
        self.goal_space = Box(np.concatenate((self.goal_low, [0, -1, -1, 0, 0, 0, 0])),
                              np.concatenate((self.goal_high, [0, 1, 1, 0, 0, 0, 0])),
                             dtype=np.float32)

        print(self.obs_space.low, self.obs_space.high)
        print(self.goal_space.low, self.goal_space.high)

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
        self.reset()

    def step(self, action):
        action = self.action_scale * action
        self.do_simulation(np.array(action))
        ob = self._get_obs()
        reward = self.compute_reward(action, ob)
        state, goal = ob['state_observation'], self._state_goal
        pos_diff = np.linalg.norm(state[:3] - goal[:3])
        angle_state, angle_goal = np.arctan2(state[3], state[4]), np.arctan2(goal[3], goal[4])
        angle_diff = np.abs(np.arctan2(np.sin(angle_state-angle_goal), np.cos(angle_state-angle_goal)))
        pos_angle_diff = np.linalg.norm(state[:5] - goal[:5])
        velocity_diff = np.linalg.norm(state[-4:-1] - goal[-4:-1])
        angular_velocity_diff = np.linalg.norm(state[-1] - goal[-1])
        info = {
            'pos_diff': pos_diff,
            'angle_diff': angle_diff,
            'pos_angle_diff': pos_angle_diff,
            'velocity_diff': velocity_diff,
            'angular_velocity_diff': angular_velocity_diff,
        }
        done = False
        return ob, reward, done, info

    def _get_obs(self):
        qpos = list(self.sim.data.qpos.flat)
        flat_obs = qpos[:-3] + [np.sin(qpos[-3]), np.cos(qpos[-3])] + list(self.sim.data.qvel.flat)[:-2]
        flat_obs = np.array(flat_obs)

        return dict(
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

    def get_goal(self):
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
        angles = np.random.uniform(0, 2 * np.pi, batch_size)
        goals[:,3], goals[:,4] = np.sin(angles), np.cos(angles)
        return {
            'desired_goal': goals,
            'state_desired_goal': goals,
        }

    def compute_rewards(self, actions, obs):
        achieved_goals = obs['state_achieved_goal']
        desired_goals = obs['state_desired_goal']
        car_pos = achieved_goals
        goals = desired_goals
        diff = car_pos - goals
        if self.reward_type == 'dense':
            r = -np.linalg.norm(diff, ord=self.norm_order, axis=1)
        elif self.reward_type == 'vectorized_dense':
            r = -np.abs(diff)
        else:
            raise NotImplementedError("Invalid/no reward type.")
        return r

    def reset_model(self):
        self._reset_car()
        self.set_goal(self.sample_goal())
        self.sim.forward()
        return self._get_obs()

    def _reset_car(self):
        qpos = np.zeros(6)
        qvel = np.zeros(6)
        qpos[0:2] = np.random.uniform(self.car_low, self.car_high)
        qpos[3] = np.random.uniform(0, 2*np.pi)
        self.set_state(qpos, qvel)

    def set_goal(self, goal):
        self._state_goal = goal['state_desired_goal']

    def set_to_goal(self, goal):
        state_goal = goal['state_desired_goal']
        qpos, qvel = np.zeros(6), np.zeros(6)
        qpos[0:3] = state_goal[0:3] #xyz pos
        qpos[3] = np.arctan2(state_goal[3], state_goal[4]) #angle
        qvel[0:3] = state_goal[-4:-1] #vel_{xyz}
        qvel[3] = state_goal[-1] #vel_angle
        self.set_state(qpos, qvel)

    def get_env_state(self):
        joint_state = self.sim.get_state()
        goal = self._state_goal.copy()
        return joint_state, goal

    def set_env_state(self, state):
        state, goal = state
        self.sim.set_state(state)
        self.sim.forward()
        self._state_goal = goal

    def get_diagnostics(self, paths, prefix=''):
        statistics = OrderedDict()
        for stat_name in [
            'pos_diff',
            'angle_diff',
            'pos_angle_diff',
            'velocity_diff',
            'angular_velocity_diff',
        ]:
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
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.lookat[0] = 0.0
        self.viewer.cam.lookat[1] = 0.0
        self.viewer.cam.lookat[2] = 0.5
        self.viewer.cam.distance = 6.5
        self.viewer.cam.elevation = -90