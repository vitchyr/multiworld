import random

import numpy as np
from gym import GoalEnv
from gym.spaces import Dict, Box

from multiworld.core.serializable import Serializable
from multiworld.envs.env_util import get_asset_full_path
from multiworld.envs.mujoco.cameras import create_camera_init
from multiworld.envs.mujoco.mujoco_env import MujocoEnv

DEFAULT_CAMERA_CONFIG = {
    'trackbodyid': 2,
    'distance': 3.0,
    'lookat': np.array((0.0, 0.0, 1.15)),
    'elevation': -20.0,
}


class HopperEnv(MujocoEnv, Serializable):
    # Copy of Hopper-v3
    def __init__(self,
                 xml_file='hopper.xml',
                 forward_reward_weight=1.0,
                 ctrl_cost_weight=1e-3,
                 healthy_reward=1.0,
                 terminate_when_unhealthy=True,
                 healthy_state_range=(-100.0, 100.0),
                 healthy_z_range=(0.7, float('inf')),
                 healthy_angle_range=(-0.2, 0.2),
                 reset_noise_scale=5e-3,
                 exclude_current_positions_from_observation=True,
                 frame_skip=4,
                 rgb_rendering_tracking=True):
        self.quick_init(locals())
        super().__init__(
            model_path=xml_file,
            frame_skip=frame_skip,
        )

        self._forward_reward_weight = forward_reward_weight

        self._ctrl_cost_weight = ctrl_cost_weight

        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy

        self._healthy_state_range = healthy_state_range
        self._healthy_z_range = healthy_z_range
        self._healthy_angle_range = healthy_angle_range

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation)

        bounds = self.model.actuator_ctrlrange.copy()
        low = bounds[:, 0]
        high = bounds[:, 1]
        self.action_space = Box(low, high)
        obs_size = self._get_flat_state_obs().shape[0]
        high = np.inf * np.ones(obs_size)
        low = -high
        self.observation_space = Box(low, high)

    @property
    def healthy_reward(self):
        return float(
            self.is_healthy
            or self._terminate_when_unhealthy
        ) * self._healthy_reward

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    @property
    def is_healthy(self):
        z, angle = self.sim.data.qpos[1:3]
        state = self.state_vector()[2:]

        min_state, max_state = self._healthy_state_range
        min_z, max_z = self._healthy_z_range
        min_angle, max_angle = self._healthy_angle_range

        healthy_state = np.all(
            np.logical_and(min_state < state, state < max_state))
        healthy_z = min_z < z < max_z
        healthy_angle = min_angle < angle < max_angle

        is_healthy = all((healthy_state, healthy_z, healthy_angle))

        return is_healthy

    @property
    def done(self):
        done = (not self.is_healthy
                if self._terminate_when_unhealthy
                else False)
        return done

    def _get_obs(self):
        return self._get_flat_state_obs()

    def _get_flat_state_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = np.clip(
            self.sim.data.qvel.flat.copy(), -10, 10)

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        observation = np.concatenate((position, velocity)).ravel()
        return observation

    def step(self, action):
        x_position_before = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.sim.data.qpos[0]
        x_velocity = ((x_position_after - x_position_before)
                      / self.dt)

        ctrl_cost = self.control_cost(action)

        forward_reward = self._forward_reward_weight * x_velocity
        healthy_reward = self.healthy_reward

        rewards = forward_reward + healthy_reward
        costs = ctrl_cost

        observation = self._get_obs()
        reward = rewards - costs
        done = self.done
        info = {
            'x_position': x_position_after,
            'x_velocity': x_velocity,
        }

        return observation, reward, done, info

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv)

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)


class HopperFullPositionGoalEnv(HopperEnv, GoalEnv, Serializable):
    def __init__(
            self,
            xml_path='classic_mujoco/hopper_full_state_goal.xml',
            presampled_positions='classic_mujoco/hopper_goal_qpos_-10to10_x_upright.npy',
            presampled_velocities='classic_mujoco/hopper_goal_qvel_-10to10_x_upright.npy',
            camera_lookat=(0, 0, 0),
            camera_distance=20,
            camera_elevation=-5,
    ):
        self.quick_init(locals())
        super().__init__(xml_file=get_asset_full_path(xml_path))
        self.goal_space = Box(
            self.observation_space.low[:6],
            self.observation_space.high[:6],
        )
        self.observation_space = Dict([
            ('observation', self.observation_space),
            ('desired_goal', self.goal_space),
            ('achieved_goal', self.goal_space),
        ])
        self.camera_init = create_camera_init(
            lookat=camera_lookat,
            distance=camera_distance,
            elevation=camera_elevation,
        )
        # self._goal = np.zeros_like(self.goal_space.low)
        # self._goal = self._get_flat_state_obs()[:6]
        self.presampled_qpos = np.load(
            get_asset_full_path(presampled_positions)
        )
        self.presampled_qvel = np.load(
            get_asset_full_path(presampled_velocities)
        )
        self._goal = None
        idx = random.randint(0, len(self.presampled_qpos)-1)
        self.goal = self.presampled_qpos[idx]

    def _get_obs(self):
        state_obs = self._get_flat_state_obs()
        return dict(
            observation=state_obs,
            desired_goal=self.goal,
            achieved_goal=self._get_achieved_goal(),
        )

    def _get_flat_state_obs(self):
        position = self.sim.data.qpos.flat.copy()[:6]
        velocity = np.clip(self.sim.data.qvel.flat.copy()[:6], -10, 10)

        observation = np.concatenate((position, velocity)).ravel()
        return observation

    def viewer_setup(self):
        self.camera_init(self.viewer.cam)

    def step(self, action):
        results = super().step(action)
        state = self.sim.get_state()
        # import ipdb; ipdb.set_trace()
        state.qpos[6:] = self.goal
        state.qvel[6:] = 0
        self.sim.set_state(state)
        return results

    def _get_achieved_goal(self):
        return self.sim.data.qpos.flat[:6].copy()

    def compute_reward(self, achieved_goal, desired_goal, info):
        return - np.linalg.norm(achieved_goal - desired_goal)

    @property
    def goal(self):
        return self._goal

    @goal.setter
    def goal(self, value):
        self._goal = value

    def _goal_site_pos(self):
        site_id = self.sim.model.site_name2id('goal')
        return self.sim.data.site_xpos[site_id]

    def reset(self):
        idx = random.randint(0, len(self.presampled_qpos)-1)
        self.goal = self.presampled_qpos[idx]
        return super().reset()
