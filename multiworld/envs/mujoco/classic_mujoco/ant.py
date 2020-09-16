"""
Exact same as gym env, except that the gear ratio is 30 rather than 150.
"""
import numpy as np
import random

from gym.spaces import Dict, Box
from gym import GoalEnv

from multiworld.core.serializable import Serializable
from multiworld.envs.mujoco.cameras import create_camera_init
from multiworld.envs.mujoco.mujoco_env import MujocoEnv
from multiworld.envs.env_util import get_asset_full_path


class AntEnv(MujocoEnv, Serializable):
    def __init__(
            self,
            use_low_gear_ratio=False,
            include_contact_forces_in_state=True,
            xml_path=None,
    ):
        self.quick_init(locals())
        if xml_path is None:
            if use_low_gear_ratio:
                xml_path = 'classic_mujoco/low_gear_ratio_ant.xml'
            else:
                xml_path = 'classic_mujoco/ant.xml'
        super().__init__(
            get_asset_full_path(xml_path),
            frame_skip=5,
        )
        self.include_contact_forces_in_state = include_contact_forces_in_state
        bounds = self.model.actuator_ctrlrange.copy()
        low = bounds[:, 0]
        high = bounds[:, 1]
        self.action_space = Box(low, high)
        obs_size = self._get_env_obs().shape[0]
        high = np.inf * np.ones(obs_size)
        low = -high
        self.observation_space = Box(low, high)

    def step(self, a):
        torso_xyz_before = self.get_body_com("torso")
        self.do_simulation(a, self.frame_skip)
        torso_xyz_after = self.get_body_com("torso")
        torso_velocity = torso_xyz_after - torso_xyz_before
        forward_reward = torso_velocity[0] / self.dt
        ctrl_cost = .5 * np.square(a).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = (
                np.isfinite(state).all()
                and 0.2 <= state[2] <= 1.0
        )
        # done = not notdone
        done = False
        ob = self._get_obs()
        return ob, reward, done, dict(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward,
            torso_velocity=torso_velocity,
        )

    def _get_obs(self):
        if self.include_contact_forces_in_state:
            return np.concatenate([
                self.sim.data.qpos.flat[2:],
                self.sim.data.qvel.flat,
                np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
            ])
        else:
            return np.concatenate([
                self.sim.data.qpos.flat[2:],
                self.sim.data.qvel.flat,
            ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq,
                                                       low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5


class AntXYGoalEnv(AntEnv, GoalEnv, Serializable):
    def __init__(self, goal_size=-5, **kwargs):
        self.quick_init(locals())
        super().__init__(**kwargs)
        low = - goal_size * np.ones(2)
        high = goal_size * np.ones(2)

        self.goal_space = Box(low, high)
        self._goal = None
        self.goal = self.goal_space.sample()
        self.observation_space = Dict([
            ('observation', self.observation_space),
            ('desired_goal', self.goal_space),
            ('achieved_goal', self.goal_space),
        ])
        self.camera_init = create_camera_init(
            lookat=(0, 0, 0),
            distance=15,
            elevation=-45,
            # trackbodyid=self.sim.model.body_name2id('torso'),
        )

    def reset(self):
        self.goal = self.goal_space.sample()
        return super().reset()

    def _get_obs(self):
        state_obs = self._get_env_obs()
        return dict(
            observation=state_obs,
            desired_goal=self.goal,
            achieved_goal=state_obs[:2],
        )

    def _get_env_obs(self):
        if self.include_contact_forces_in_state:
            return np.concatenate([
                self.sim.data.qpos.flat,
                self.sim.data.qvel.flat,
                np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
            ])
        else:
            return np.concatenate([
                self.sim.data.qpos.flat,
                self.sim.data.qvel.flat,
            ])

    def compute_reward(self, achieved_goal, desired_goal, info):
        return - np.linalg.norm(achieved_goal - desired_goal)

    @property
    def goal(self):
        return self._goal

    @goal.setter
    def goal(self, value):
        self._goal = value
        site_id = self.sim.model.site_name2id('goal')
        self.sim.model.site_pos[site_id] = np.concatenate(
            (self._goal, np.zeros(1)),
            axis=0,
        )

    def _goal_site_pos(self):
        site_id = self.sim.model.site_name2id('goal')
        return self.sim.data.site_xpos[site_id]

    def viewer_setup(self):
        self.camera_init(self.viewer.cam)


class AntFullPositionGoalEnv(AntEnv, GoalEnv, Serializable):
    def __init__(
            self,
            presampled_positions='classic_mujoco/ant_goal_qpos_5x5_xy.npy',
    ):
        self.quick_init(locals())
        super().__init__(
            include_contact_forces_in_state=False,
            xml_path='classic_mujoco/ant_full_state_goal.xml',
        )
        self.goal_space = Box(
            self.observation_space.low[:15],
            self.observation_space.high[:15],
        )
        self.observation_space = Dict([
            ('observation', self.observation_space),
            ('desired_goal', self.goal_space),
            ('achieved_goal', self.goal_space),
        ])
        self.camera_init = create_camera_init(
            lookat=(0, 0, 0),
            distance=15,
            elevation=-45,
            # trackbodyid=self.sim.model.body_name2id('torso'),
        )
        self.presampled_qpos = np.load(
            get_asset_full_path(presampled_positions)
        )
        self._goal = None
        idx = random.randint(0, len(self.presampled_qpos)-1)
        self.goal = self.presampled_qpos[idx]

    def reset(self):
        idx = random.randint(0, len(self.presampled_qpos)-1)
        self.goal = self.presampled_qpos[idx]
        return super().reset()

    def _get_obs(self):
        state_obs = self._get_env_obs()
        return dict(
            observation=state_obs,
            desired_goal=self.goal,
            achieved_goal=self._get_achieved_goal(),
        )

    def step(self, action):
        results = super().step(action)
        state = self.sim.get_state()
        state.qpos[15:] = self.goal
        state.qvel[14:] = 0
        self.sim.set_state(state)
        return results

    def _get_achieved_goal(self):
        return self.sim.data.qpos.flat[:15]

    def _get_env_obs(self):
        if self.include_contact_forces_in_state:
            return np.concatenate([
                self.sim.data.qpos.flat[:15],
                self.sim.data.qvel.flat[:15],
                np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
            ])
        else:
            return np.concatenate([
                self.sim.data.qpos.flat[:15],
                self.sim.data.qvel.flat[:15],
            ])

    def compute_reward(self, achieved_goal, desired_goal, info):
        return - np.linalg.norm(achieved_goal - desired_goal)

    @property
    def goal(self):
        return self._goal

    @goal.setter
    def goal(self, value):
        # value = self.sim.get_state().qpos[:15].copy()
        # value[0] += 1
        # import ipdb; ipdb.set_trace()
        self._goal = value
        # value = self.sim.model.geom_pos[1:14].copy()
        # value[:, 0] += 1
        # value[:, 1] += 1
        # self.sim.model.geom_pos[14:27] = value
        # import ipdb; ipdb.set_trace()
        # site_id = self.sim.model.site_name2id('goal')
        # self.sim.model.site_pos[site_id] = np.concatenate(
        #     (self._goal, np.zeros(1)),
        #     axis=0,
        # )

    def _goal_site_pos(self):
        site_id = self.sim.model.site_name2id('goal')
        return self.sim.data.site_xpos[site_id]

    def viewer_setup(self):
        self.camera_init(self.viewer.cam)
