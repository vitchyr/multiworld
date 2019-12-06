import os

from gym import error, spaces
from gym.utils import seeding
import numpy as np
from os import path
import gym

from multiworld.core.multitask_env import MultitaskEnv
from multiworld.core.serializable import Serializable
from multiworld.envs.env_util import (
    get_stat_in_paths,
    create_stats_ordered_dict,
)
from collections import OrderedDict
import logging

from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig, CameraConfig
from rlbench.tasks import ReachTarget
import numpy as np

import signal
import sys

class RLBenchEnv(MultitaskEnv, Serializable):
    """
    Multiworld wrapper for RLBench environments
    """
    sim = None
    task = None
    action_dim = None

    def __init__(
            self,
            task_class,
            fixed_goal=None,
            headless=False,
            camera=False, # False or image_size
            stub=False,
            state_observation_type="task",
            **kwargs
    ):
        self.quick_init(locals())

        self.task_class = task_class
        if fixed_goal is not None:
            fixed_goal = np.array(fixed_goal)
        self.fixed_goal = fixed_goal
        self.state_observation_type = state_observation_type

        if not stub and RLBenchEnv.sim == None: # only launch sim once
            obs_config = ObservationConfig()
            obs_config.set_all(False)
            if camera:
                camera_config = CameraConfig(image_size=camera)
                obs_config.left_shoulder_camera = camera_config
            obs_config.set_all_low_dim(True)
            # obs_config.right_shoulder_camera.rgb = True

            action_mode = ActionMode(ArmActionMode.ABS_JOINT_VELOCITY)
            RLBenchEnv.action_dim = action_mode.action_size

            RLBenchEnv.sim = Environment(
                action_mode, obs_config=obs_config, headless=headless,
                # static_positions=True,
            )
            RLBenchEnv.sim.launch()

            RLBenchEnv.task = RLBenchEnv.sim.get_task(self.task_class)

            # Add a signal handler to gracefully shut down on Ctrl-C or kill
            signal.signal(signal.SIGINT, self.signal_handler)
            signal.signal(signal.SIGTERM, self.signal_handler)

            self.reset()


        if stub:
            self.task = None
            u = np.ones((8, ))
        else:
            self.task = RLBenchEnv.task
            u = np.ones(RLBenchEnv.action_dim)

        self.action_space = spaces.Box(-u, u, dtype=np.float32)

        if self.state_observation_type == "task":
            OBS_DIM = 101 + 7 + 7
        elif self.state_observation_type == "joints":
            OBS_DIM = 7 + 7
        else:
            error

        self._target_position = None
        self._position = np.zeros((OBS_DIM, ))
        self.boundary_dist = 5
        o = self.boundary_dist * np.ones((OBS_DIM, ))
        self.obs_range = spaces.Box(-o, o, dtype='float32')
        self.goal_space = spaces.Box(
            np.zeros((0, )),
            np.zeros((0, )),
            dtype=np.float32,
        )
        self.observation_space = spaces.Dict([
            ('observation', self.obs_range),
            ('desired_goal', self.goal_space),
            ('achieved_goal', self.goal_space),
            ('state_observation', self.obs_range),
            ('state_desired_goal', self.goal_space),
            ('state_achieved_goal', self.goal_space),
        ])

        self.drawer = None
        self.render_drawer = None

    def signal_handler(self, sig, frame):
        print('You pressed Ctrl+C!')
        RLBenchEnv.sim.shutdown()
        sys.exit(0)

    def step(self, velocities):
        u = velocities.copy()
        u[-1] = float(u[-1] > 0.5)
        ob, reward, done = self.task.step(u)
        self.ob = ob
        info = dict(
            task_reward=reward,
        )
        obs = self._get_obs()
        return obs, reward, done, info

    def reset(self):
        descriptions, ob = self.task.reset()
        self.ob = ob
        self._target_position = self.sample_goal()['state_desired_goal']
        # if self.randomize_position_on_reset:
        #     self._position = self._sample_position(
        #         self.obs_range.low,
        #         self.obs_range.high,
        #     )
        return self._get_obs()

    def _get_obs(self):
        # state = self.ob.task_low_dim_state
        # state = self.ob.joint_positions
        s1 = self.ob.task_low_dim_state
        s2 = self.ob.gripper_pose
        s3 = self.ob.joint_positions
        if self.state_observation_type == "task":
            state = np.concatenate((s1, s2, s3))
        elif self.state_observation_type == "joints":
            state = np.concatenate((s2, s3))
        else:
            error
        return dict(
            observation=state.copy(),
            desired_goal=self._target_position.copy(),
            achieved_goal=self._target_position.copy(),
            state_observation=state.copy(),
            state_desired_goal=self._target_position.copy(),
            state_achieved_goal=self._target_position.copy(),
        )

    def compute_rewards(self, actions, obs, info=None):
        achieved_goals = obs['state_achieved_goal']
        desired_goals = obs['state_desired_goal']
        d = np.linalg.norm(achieved_goals - desired_goals, axis=-1)
        # if self.reward_type == "sparse":
        #     return -(d > self.target_radius).astype(np.float32)
        # elif self.reward_type == "dense":
        #     return -d
        # elif self.reward_type == 'vectorized_dense':
        #     return -np.abs(achieved_goals - desired_goals)
        # else:
        #     raise NotImplementedError()
        return -d

    def get_diagnostics(self, paths, prefix=''):
        statistics = OrderedDict()
        # for stat_name in [
        #     'radius',
        #     'target_position',
        #     'distance_to_target',
        #     'velocity',
        #     'speed',
        #     'is_success',
        # ]:
        #     stat_name = stat_name
        #     stat = get_stat_in_paths(paths, 'env_infos', stat_name)
        #     statistics.update(create_stats_ordered_dict(
        #         '%s%s' % (prefix, stat_name),
        #         stat,
        #         always_show_all_stats=True,
        #         ))
        #     statistics.update(create_stats_ordered_dict(
        #         'Final %s%s' % (prefix, stat_name),
        #         [s[-1] for s in stat],
        #         always_show_all_stats=True,
        #         ))
        return statistics

    def get_goal(self):
        return {
            'desired_goal': self._target_position.copy(),
            'state_desired_goal': self._target_position.copy(),
        }

    def sample_goals(self, batch_size):
        if not self.fixed_goal is None:
            goals = np.repeat(
                self.fixed_goal.copy()[None],
                batch_size,
                0
            )
        else:
            goals = np.zeros((batch_size, self.obs_range.low.size))
            for b in range(batch_size):
                if batch_size > 1:
                    logging.warning("This is very slow!")
                goals[b, :] = self._sample_position(
                    self.obs_range.low,
                    self.obs_range.high,
                )
        return {
            'desired_goal': goals,
            'state_desired_goal': goals,
        }

    def set_position(self, pos):
        # self._position[0] = pos[0]
        # self._position[1] = pos[1]
        pass

    """Functions for ImageEnv wrapper"""

    def get_image(self, width=None, height=None):
        """Returns a black and white image"""
        return np.uint8(self.ob.left_shoulder_rgb * 255)

    def set_to_goal(self, goal_dict):
        goal = goal_dict["state_desired_goal"]
        # self._position = goal
        # self._target_position = goal

    def set_goal(self, goal_dict):
        pass

    def get_env_state(self):
        return self._get_obs()

    def set_env_state(self, state):
        position = state["state_observation"]
        goal = state["state_desired_goal"]
        # self._position = position
        # self._target_position = goal

    def initialize_camera(self, init_fctn):
        pass
