import abc
import numpy as np
from gym.spaces import Box, Dict

from multiworld.core.serializable import Serializable
from multiworld.envs.mujoco.mujoco_env import MujocoEnv
from multiworld.core.multitask_env import MultitaskEnv
from multiworld.envs.env_util import get_asset_full_path


class WheeledEnv(MujocoEnv, Serializable, MultitaskEnv, metaclass=abc.ABCMeta):
    def __init__(
            self,
            reward_type='dense',
            norm_order=2,

            frame_skip=3,
            *args,
            **kwargs):
        self.quick_init(locals())
        MultitaskEnv.__init__(self)
        MujocoEnv.__init__(self,
                           model_path=get_asset_full_path('locomotion/wheeled.xml'),
                           frame_skip=frame_skip,
                           **kwargs)

        self.action_space = Box(np.array([-1, -1]), np.array([1, 1]), dtype=np.float32)

        self.reward_type = reward_type
        self.norm_order = norm_order

        self.obs_space = Box(np.array([-1, -1]), np.array([1, 1]), dtype=np.float32)
        self.goal_space = Box(np.array([-2, -2]), np.array([2, 2]), dtype=np.float32)

        self.observation_space = Dict([
            ('observation', self.obs_space),
            ('desired_goal', self.goal_space),
            ('achieved_goal', self.goal_space),
            ('state_observation', self.obs_space),
            ('state_desired_goal', self.goal_space),
            ('state_achieved_goal', self.goal_space),
            ('proprio_observation', self.obs_space),
            ('proprio_desired_goal', self.goal_space),
            ('proprio_achieved_goal', self.goal_space),
        ])

        self._state_goal = None
        self.reset()

    def step(self, action):
        self.do_simulation(np.array(action))
        ob = self._get_obs()
        reward = self.compute_reward(action, ob)
        done = False
        return ob, reward, done, {}

    def _get_obs(self):
        qpos = list(self.sim.data.qpos.flat)
        flat_obs = qpos[:-3] + [np.sin(qpos[-3]), np.cos(qpos[-3])] + list(self.sim.data.qvel.flat)[:-2]
        flat_obs = np.array(flat_obs)

        return dict(
            observation=flat_obs,
            desired_goal=self._state_goal,
            achieved_goal=flat_obs[:2],
            state_observation=flat_obs,
            state_desired_goal=self._state_goal,
            state_achieved_goal=flat_obs[:2],
            proprio_observation=flat_obs,
            proprio_desired_goal=self._state_goal,
            proprio_achieved_goal=flat_obs[:2],
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
        qpos[0] = 1.5
        qpos[1] = 1.5
        # qpos[0] = np.random.uniform(-1.8, 1.8)
        # qpos[1] = np.random.uniform(-1.8, 1.8)
        qpos[3] = np.random.uniform(0, 2*np.pi)

        qvel = np.zeros(6)
        self.set_state(qpos, qvel)

    def set_goal(self, goal):
        self._state_goal = goal['state_desired_goal']

    # def reset(self, reset_args=None, init_state=None, **kwargs):
    #     # body_pos = self.model.body_pos.copy()
    #     # print(body_pos)
    #     # body_pos[-1][:2] = self.goal
    #     # self.model.body_pos = body_pos
    #     self.model.forward()
    #     self.current_com = self.model.data.com_subtree[0]
    #     self.dcom = np.zeros_like(self.current_com)
    #     obs = self._get_obs()
    #     return obs

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.lookat[0] = 0.0
        self.viewer.cam.lookat[1] = 0.0
        self.viewer.cam.lookat[2] = 0.5
        self.viewer.cam.distance = 6.0
        self.viewer.cam.elevation = -90
        # self.viewer.cam.azimuth = 270
        # self.viewer.cam.trackbodyid = -1