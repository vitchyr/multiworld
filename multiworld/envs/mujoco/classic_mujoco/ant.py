import abc
import copy
import numpy as np
from gym.spaces import Box, Dict

from multiworld.core.serializable import Serializable
from multiworld.envs.mujoco.mujoco_env import MujocoEnv
from multiworld.core.multitask_env import MultitaskEnv
from multiworld.envs.env_util import get_asset_full_path
import os.path as osp

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from collections import OrderedDict

from multiworld.envs.env_util import get_stat_in_paths, create_stats_ordered_dict

DEFAULT_GOAL = [-2., -2., 0.565, 1., 0., 0., 0., 0.,
                1., 0., -1., 0., -1., 0., 1., -3.,
                -3., 0.75, 1., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0.]

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
            init_qpos=list([
                -2, -2, 0.565, 1,
                0, 0, 0,
                0, 1., 0., -1., 0., -1., 0., 1.,
            ]),
            fixed_goal=None, # deprecated feature
            init_xy_mode='fixed',
            terminate_when_unhealthy=False,
            healthy_z_range=(0.2, 0.9),
            health_reward=10,
            goal_sampling_strategy='uniform',
            presampled_goal_paths='',
            fixed_goal_qpos=None,
            test_mode_case_num=None,
            use_euler=False,
            reset_low=None,
            reset_high=None,
            reset_and_goal_mode=None,

            v_func_heatmap_bounds=(-2.5, 0.0),
            wall_collision_buffer=0.0,

            sparsity_threshold=1.5,
            *args,
            **kwargs):
        assert init_xy_mode in {
            'fixed',
            'uniform',
            'uniform_pos_and_rot',

            'sample-uniformly-xy-space',
        }

        assert reset_and_goal_mode in {
            'fixed',
            'uniform',
            'uniform_pos_and_rot',
            None
        }

        if reset_and_goal_mode is not None:
            init_xy_mode = reset_and_goal_mode
            goal_sampling_strategy = reset_and_goal_mode

        if model_path == 'classic_mujoco/ant_maze2_gear30_small_dt3.xml':
            model_path = 'classic_mujoco/ant_gear30_dt3_u_small.xml'
        elif model_path == 'classic_mujoco/ant_maze2_gear30_big_dt3.xml':
            model_path = 'classic_mujoco/ant_gear30_dt3_u_big.xml'
        elif model_path == 'classic_mujoco/ant_fb_gear30_med_dt3.xml':
            model_path = 'classic_mujoco/ant_gear30_dt3_fb_med.xml'

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
        self.sparsity_threshold = sparsity_threshold
        self.norm_order = norm_order
        self.goal_is_xy = goal_is_xy
        self.goal_is_qpos = goal_is_qpos
        self.fixed_goal_qpos = fixed_goal_qpos
        self.init_xy_mode = init_xy_mode
        self.terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_reward = health_reward
        self._healthy_z_range = healthy_z_range
        # self.ant_radius = 0.75

        self.model_path = model_path
        assert goal_sampling_strategy in {
            'fixed',
            'uniform',
            'uniform_pos_and_rot',

            'presampled'
        }
        self.goal_sampling_strategy = goal_sampling_strategy
        if self.goal_sampling_strategy == 'presampled':
            assert presampled_goal_paths is not None
            from railrl.misc.asset_loader import load_local_or_remote_file
            self.presampled_goals = load_local_or_remote_file(presampled_goal_paths)
        else:
            self.presampled_goals = None

        self.use_euler = use_euler

        self.ant_low, self.ant_high = np.array(ant_low), np.array(ant_high)

        goal_low, goal_high = np.array(goal_low), np.array(goal_high)
        self.goal_low, self.goal_high = goal_low, goal_high

        if reset_low is not None:
            reset_low = np.array(reset_low)
        else:
            reset_low = goal_low.copy()
        if reset_high is not None:
            reset_high = np.array(reset_high)
        else:
            reset_high = goal_high.copy()
        self.reset_low, self.reset_high = reset_low, reset_high

        self.two_frames = two_frames
        self.vel_in_state = vel_in_state

        if self.use_euler:
            self.pos_dim = 17
        else:
            self.pos_dim = 15

        if self.vel_in_state:
            self.state_dim = self.pos_dim + 14
        else:
            self.state_dim = self.pos_dim

        obs_space_low = np.concatenate((self.ant_low, -np.ones(self.state_dim-2)))
        obs_space_high = np.concatenate((self.ant_high, np.ones(self.state_dim-2)))
        if goal_is_xy:
            goal_space_low = goal_low
            goal_space_high = goal_high
        else:
            goal_space_low = np.concatenate((goal_low, -np.ones(self.state_dim-2)))
            goal_space_high = np.concatenate((goal_high, np.ones(self.state_dim-2)))

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

        spaces = [
            ('observation', self.obs_space),
            ('desired_goal', self.goal_space),
            ('achieved_goal', self.goal_space),
            ('state_observation', self.obs_space),
            ('state_desired_goal', self.goal_space),
            ('state_achieved_goal', self.goal_space),
        ]
        self.observation_space = Dict(spaces)

        self._full_state_goal = None
        self._prev_obs = None
        self._cur_obs = None
        self.subgoals = None
        self.test_mode_case_num = test_mode_case_num
        self.timesteps_so_far = 0
        self.tau = None

        self.v_func_heatmap_bounds = v_func_heatmap_bounds
        self.wall_collision_buffer = wall_collision_buffer

        self.sweep_goal = None

        # self.reset()

    def step(self, action):
        self._prev_obs = self._cur_obs
        flipped_before_step = self.is_flipped
        self.do_simulation(np.array(action), self.frame_skip)
        self.timesteps_so_far += 1
        ob = self._get_obs()
        reward = self.compute_reward(action, ob)
        info = {}
        # if self._full_state_goal is not None:
        info['full-state-distance'] = self._compute_state_distances(
            self.numpy_batchify_dict(ob)
        )
        info['qpos-distance'] = self._compute_qpos_distances(
            self.numpy_batchify_dict(ob)
        )
        info['epos-distance'] = self._compute_epos_distances(
            self.numpy_batchify_dict(ob)
        )
        info['epos-sparse'] = self._compute_epos_distances(
            self.numpy_batchify_dict(ob),
            vectorized=False, sparse=True,
        )
        info['quat-distance'] = self._compute_quat_distances(
            self.numpy_batchify_dict(ob)
        )
        info['euler-distance'] = self._compute_euler_distances(
            self.numpy_batchify_dict(ob)
        )
        info['xy-distance'] = self._compute_xy_distances(
            self.numpy_batchify_dict(ob)
        )
        info['xy-success'] = self._compute_xy_successes(
            self.numpy_batchify_dict(ob)
        )
        info['xy-success2'] = self._compute_xy_successes2(
            self.numpy_batchify_dict(ob)
        )
        info['xy-success3'] = self._compute_xy_successes3(
            self.numpy_batchify_dict(ob)
        )
        info['xy-success4'] = self._compute_xy_successes4(
            self.numpy_batchify_dict(ob)
        )
        info['leg-distance'] = self._compute_leg_distances(
            self.numpy_batchify_dict(ob)
        )
        if self.terminate_when_unhealthy:
            done = not self.is_healthy
            reward += self._healthy_reward
        else:
            done = False

        if not flipped_before_step and self.is_flipped:
            info['flip_time'] = self.timesteps_so_far
            if self.tau is not None:
                info['flip_tau'] = self.tau

        info['is_flipped'] = int(self.is_flipped)
        info['is_not_healthy'] = int(not self.is_healthy)

        return ob, reward, done, info

    def set_tau(self, tau):
        self.tau = tau

    def get_diagnostics(self, paths, prefix=''):
        statistics = OrderedDict()
        for stat_name in [
            'full-state-distance',
            'qpos-distance',
            'epos-distance',
            'epos-sparse',
            'quat-distance',
            'euler-distance',
            'xy-distance',
            'xy-success',
            'xy-success2',
            'xy-success3',
            'xy-success4',
            'leg-distance',
            'is_not_healthy',
            'is_flipped',
        ]:
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

        for stat_name in [
            'flip_time',
            'flip_tau',
        ]:
            stat = []
            for path in paths:
                for info in path['env_infos']:
                    if stat_name in info:
                        stat.append(info[stat_name])
            if len(stat) == 0:
                stat = [-1]
            statistics.update(create_stats_ordered_dict(
                '%s%s' % (prefix, stat_name),
                stat,
                always_show_all_stats=True,
            ))

        return statistics

    @property
    def is_healthy(self):
        state = self.state_vector()
        min_z, max_z = self._healthy_z_range
        is_healthy = (np.isfinite(state).all() and min_z <= state[2] <= max_z)
        return is_healthy

    @property
    def is_flipped(self):
        state = self.state_vector()
        euler = self._qpos_to_epos(state[:15])[3:9]
        pitch = np.arctan2(euler[3], euler[2])
        yaw = np.arctan2(euler[5], euler[4])
        return np.abs(np.degrees(pitch)) > 90 or np.abs(np.degrees(yaw)) > 90

    def _check_flipped(self, euler):
        pitch = np.arctan2(euler[:,3], euler[:,2])
        yaw = np.arctan2(euler[:,5], euler[:,4])
        return np.logical_or(np.abs(np.degrees(pitch)) > 90, np.abs(np.degrees(yaw)) > 90)

    def _quat_to_euler(self, quat):
        x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        X = np.arctan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = np.clip(t2, -1.0, 1.0)
        Y = np.arcsin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        Z = np.arctan2(t3, t4)

        return np.column_stack((
            np.cos(X), np.sin(X),
            np.cos(Y), np.sin(Y),
            np.cos(Z), np.sin(Z),
        ))

    def get_image_v(self, agent, qf, vf, obs, tau=None, imsize=None):
        nx, ny = (50, 50)
        sweep_obs = np.tile(obs.reshape((1, -1)), (nx * ny, 1))

        if self.sweep_goal is None:
            if self.model_path in [
                'classic_mujoco/ant_maze2_gear30_small_dt3.xml',
                'classic_mujoco/ant_gear30_dt3_u_small.xml',
            ]:
                extent = [-3.5, 3.5, -3.5, 3.5]
            elif self.model_path in [
                'classic_mujoco/ant_gear30_dt3_u_med.xml',
                'classic_mujoco/ant_gear15_dt3_u_med.xml',
                'classic_mujoco/ant_gear10_dt3_u_med.xml',
                'classic_mujoco/ant_gear30_dt2_u_med.xml',
                'classic_mujoco/ant_gear15_dt2_u_med.xml',
                'classic_mujoco/ant_gear10_dt2_u_med.xml',
                'classic_mujoco/ant_gear30_u_med.xml',
            ]:
                extent = [-4.5, 4.5, -4.5, 4.5]
            elif self.model_path in [
                'classic_mujoco/ant_fb_gear30_med_dt3.xml',
                'classic_mujoco/ant_gear30_dt3_fb_med.xml',
                'classic_mujoco/ant_gear15_dt3_fb_med.xml',
                'classic_mujoco/ant_gear10_dt3_fb_med.xml',
            ]:
                extent = [-6.0, 6.0, -6.0, 6.0]
            elif self.model_path in [
                'classic_mujoco/ant_gear10_dt3_u_long.xml',
                'classic_mujoco/ant_gear15_dt3_u_long.xml',
            ]:
                extent = [-3.75, 3.75, -9.0, 9.0]
            elif self.model_path in [
                'classic_mujoco/ant_gear10_dt3_maze_med.xml',
            ]:
                extent = [-8.0, 8.0, -8.0, 8.0]
            elif self.model_path in [
                'classic_mujoco/ant_gear10_dt3_fg_med.xml',
            ]:
                extent = [-8.0, 8.0, -8.0, 8.0]
            else:
                extent = [-5.5, 5.5, -5.5, 5.5]

            x = np.linspace(extent[0], extent[1], nx)
            y = np.linspace(extent[2], extent[3], ny)
            xv, yv = np.meshgrid(x, y)

            sweep_goal_xy = np.stack((xv, yv), axis=2).reshape((-1, 2))
            init_epos = self._qpos_to_epos(self.init_qpos)
            sweep_goal_rest = np.tile(np.concatenate((init_epos[2:], np.zeros(14)))[None], (nx*ny, 1))
            sweep_goal = np.concatenate((sweep_goal_xy, sweep_goal_rest), axis=1)
            self.sweep_goal = sweep_goal
        else:
            sweep_goal = self.sweep_goal

        if tau is not None:
            sweep_tau = np.tile(tau, (nx * ny, 1))
        if vf is not None:
            if tau is not None:
                v_vals = vf.eval_np(sweep_obs, sweep_goal, sweep_tau)
            else:
                sweep_obs_goal = np.hstack((sweep_obs, sweep_goal))
                v_vals = vf.eval_np(sweep_obs_goal)
        else:
            if tau is not None:
                # sweep_actions = agent.eval_np(sweep_obs, sweep_goal, sweep_tau)
                sweep_actions = agent.get_actions(sweep_obs, sweep_goal, sweep_tau)
                v_vals = qf.eval_np(sweep_obs, sweep_actions, sweep_goal, sweep_tau)
            else:
                sweep_obs_goal = np.hstack((sweep_obs, sweep_goal))
                # sweep_actions = agent.eval_np(sweep_obs_goal)
                sweep_actions = agent.get_actions(sweep_obs_goal)
                v_vals = qf.eval_np(sweep_obs_goal, sweep_actions)
        if self.v_func_heatmap_bounds is not None:
            v_vals = v_vals / agent.reward_scale
        if v_vals.ndim == 2:
            if hasattr(qf, "norm_order"):
                norm_order = qf.norm_order
            else:
                norm_order = 2
            v_vals = -np.linalg.norm(v_vals, ord=norm_order, axis=1)
        v_vals = v_vals.reshape((nx, ny))
        if self.v_func_heatmap_bounds is not None:
            vmin = self.v_func_heatmap_bounds[0]
            vmax = self.v_func_heatmap_bounds[1]
        else:
            vmin, vmax = None, None
        return self.get_image_plt(
            v_vals,
            draw_walls=True,
            draw_state=True,
            draw_goal=True,
            draw_subgoals=True,
            vmin=vmin, vmax=vmax,
            imsize=imsize,
        )

    def _euler_to_quat(self, euler):
        roll = np.arctan2(euler[:, 1], euler[:, 0])
        pitch = np.arctan2(euler[:, 3], euler[:, 2])
        yaw = np.arctan2(euler[:, 5], euler[:, 4])

        qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(
            yaw / 2)
        qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(
            yaw / 2)
        qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(
            yaw / 2)
        qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(
            yaw / 2)

        return np.column_stack((qx, qy, qz, qw))

    def _qpos_to_epos(self, qpos):
        qpos = np.array(qpos)
        ndim = qpos.ndim
        qpos = qpos.reshape(-1, 15)
        quat = qpos[:, 3:7]
        euler = self._quat_to_euler(quat)

        epos = np.hstack((qpos[:,0:3], euler, qpos[:,7:]))

        if ndim == 1:
            return epos[0]
        else:
            return epos

    def _epos_to_qpos(self, epos):
        epos = np.array(epos)
        ndim = epos.ndim
        epos = epos.reshape(-1, 17)
        euler = epos[:, 3:9]
        quat = self._euler_to_quat(euler)

        qpos = np.hstack((
            epos[:,0:3],
            quat,
            epos[:,9:]
        ))

        if ndim == 1:
            return qpos[0]
        else:
            return qpos

    def _get_obs(self):
        qpos = list(self.sim.data.qpos.flat)[:15]
        if self.use_euler:
            epos = self._qpos_to_epos(qpos)
            flat_obs = list(epos)
        else:
            flat_obs = qpos

        if self.vel_in_state:
            flat_obs = flat_obs + list(self.sim.data.qvel.flat[:14])
        flat_obs = np.array(flat_obs)

        self._cur_obs = dict(
            observation=flat_obs,
            desired_goal=self._full_state_goal,
            achieved_goal=flat_obs,
            state_observation=flat_obs,
            state_desired_goal=self._full_state_goal,
            state_achieved_goal=flat_obs,
        )

        if self.two_frames:
            if self._prev_obs is None:
                self._prev_obs = self._cur_obs
            ob = self.merge_frames(self._prev_obs, self._cur_obs)
        else:
            ob = self._cur_obs

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
            }
        else:
            goal_dict = {
                'desired_goal': self._full_state_goal,
                'state_desired_goal': self._full_state_goal,
            }
            copied_goal_dict = {}
            for k, v in goal_dict.items():
                if goal_dict[k] is not None:
                    copied_goal_dict[k] = v.copy()
                else:
                    copied_goal_dict[k] = v
            return copied_goal_dict

    def sample_goals(self, batch_size, mode=None, keys=None):
        # if mode == 'top_left':
        #     assert not self.use_euler
        #     qpos = self.init_qpos.copy().reshape(1, -1)
        #     qpos = np.tile(qpos, (batch_size, 1))
        #     if 'small' in self.model_path:
        #         qpos[:,:2] = np.random.uniform(
        #             [-2.5, 2.5],
        #             [-2.25, 2.5],
        #             size=(batch_size, 2),
        #         )
        #     elif 'big' in self.model_path:
        #         qpos[:,:2] = np.random.uniform(
        #             [-4.25, 4.25],
        #             [-3.25, 4.25],
        #             size=(batch_size, 2),
        #         )
        #     else:
        #         raise NotImplementedError
        #
        #     if self.vel_in_state:
        #         qvel = np.zeros((batch_size, 14))
        #         state_goals = np.concatenate((qpos, qvel), axis=1)
        # elif mode == 'top_right':
        #     assert not self.use_euler
        #     qpos = self.init_qpos.copy().reshape(1, -1)
        #     qpos = np.tile(qpos, (batch_size, 1))
        #     if 'small' in self.model_path:
        #         qpos[:,:2] = np.random.uniform(
        #             [2.25, 2.5],
        #             [2.5, 2.5],
        #             size=(batch_size, 2),
        #         )
        #     elif 'big' in self.model_path:
        #         qpos[:, :2] = np.random.uniform(
        #             [3.25, 4.25],
        #             [4.25, 4.25],
        #             size=(batch_size, 2),
        #         )
        #     else:
        #         raise NotImplementedError
        #
        #     if self.vel_in_state:
        #         qvel = np.zeros((batch_size, 14))
        #         state_goals = np.concatenate((qpos, qvel), axis=1)
        if self.fixed_goal_qpos is not None:
            assert not self.use_euler
            fixed_goal = self.fixed_goal_qpos
            if self.vel_in_state:
                fixed_goal = np.concatenate((fixed_goal, np.zeros(14)))
            state_goals = np.tile(fixed_goal, (batch_size, 1))
        elif self.goal_sampling_strategy == 'uniform':
            qpos = self.init_qpos.copy().reshape(1, -1)
            qpos = np.tile(qpos, (batch_size, 1))
            qpos[:,:2] = self._sample_uniform_xy(batch_size, mode='goal')

            if self.use_euler:
                pos = self._qpos_to_epos(qpos)
            else:
                pos = qpos

            if self.vel_in_state:
                qvel = np.zeros((batch_size, 14))
                state_goals = np.concatenate((pos, qvel), axis=1)
            else:
                state_goals = pos
        elif self.goal_sampling_strategy == 'uniform_pos_and_rot':
            qpos = self.init_qpos.copy().reshape(1, -1)
            qpos = np.tile(qpos, (batch_size, 1))
            qpos[:,:2] = self._sample_uniform_xy(batch_size, mode='goal')

            rots = np.random.randint(4, size=batch_size)
            for i in range(batch_size):
                if rots[i] == 0:
                    qpos[i,3:7] = [1, 0, 0, 0]
                elif rots[i] == 1:
                    qpos[i, 3:7] = [0, 0, 0, 1]
                elif rots[i] == 2:
                    qpos[i, 3:7] = [0.7071068, 0, 0, 0.7071068]
                elif rots[i] == 3:
                    qpos[i, 3:7] = [0.7071068, 0, 0, -0.7071068]

            if self.use_euler:
                pos = self._qpos_to_epos(qpos)
            else:
                pos = qpos

            if self.vel_in_state:
                qvel = np.zeros((batch_size, 14))
                state_goals = np.concatenate((pos, qvel), axis=1)
        elif self.goal_sampling_strategy == 'presampled':
            idxs = np.random.randint(
                self.presampled_goals.shape[0], size=batch_size,
            )
            state_goals = self.presampled_goals[idxs, :]
            if self.use_euler:
                qpos = state_goals[:,:15]
                qvel = state_goals[:,15:]
                epos = self._qpos_to_epos(qpos)
                state_goals = np.hstack((epos, qvel))
            if not self.vel_in_state:
                state_goals = state_goals[:, :self.pos_dim]
        else:
            raise NotImplementedError(self.goal_sampling_strategy)

        if self.two_frames:
            state_goals = np.concatenate((state_goals, state_goals), axis=1)

        goals_dict = {
            'desired_goal': state_goals.copy(),
            'state_desired_goal': state_goals.copy(),
        }

        return goals_dict

    def _sample_uniform_xy(self, batch_size, mode='goal'):
        assert mode in ['reset', 'goal']

        if mode == 'reset':
            low, high = self.reset_low, self.reset_high
        elif mode == 'goal':
            low, high = self.goal_low, self.goal_high

        goals = np.random.uniform(
            low,
            high,
            size=(batch_size, 2),
        )
        return goals

    def compute_rewards(self, actions, obs, prev_obs=None):
        if self.reward_type in ['xy_dense', 'xy']:
            r = - self._compute_xy_distances(obs)
        elif self.reward_type in ['state', 'dense']:
            r = - self._compute_state_distances(obs)
        elif self.reward_type in ['qpos_dense', 'qpos']:
            r = - self._compute_qpos_distances(obs)
        elif self.reward_type in ['epos_dense', 'epos']:
            r = - self._compute_epos_distances(obs)
        elif self.reward_type in ['epos_sparse']:
            r = - self._compute_epos_distances(obs, vectorized=False, sparse=True)
        elif self.reward_type in ['vectorized_epos']:
            r = - self._compute_epos_distances(obs, vectorized=True)
        elif self.reward_type == 'epos_weighted_euler':
            r = - self._compute_epos_weighted_euler_distances(obs)
        elif self.reward_type == 'epos_penalize_flip':
            r = - self._compute_epos_penalize_flip_distances(obs)
        elif self.reward_type in ['state_vectorized', 'vectorized_dense']:
            r = - self._compute_vectorized_state_distances(obs)
        else:
            raise NotImplementedError("Invalid/no reward type.")
        return r

    def _compute_state_distances(self, obs):
        achieved_goals = obs['state_achieved_goal']
        desired_goals = obs['state_desired_goal']
        if desired_goals.shape == (1,):
            return -1000
        ant_pos = achieved_goals
        goals = desired_goals
        diff = ant_pos - goals
        return np.linalg.norm(diff, ord=self.norm_order, axis=1)

    def _compute_vectorized_state_distances(self, obs):
        achieved_goals = obs['state_achieved_goal']
        desired_goals = obs['state_desired_goal']
        ant_pos = achieved_goals
        goals = desired_goals
        diff = ant_pos - goals
        return np.abs(diff)

    def _compute_xy_distances(self, obs):
        if self.two_frames:
            state_size = obs['state_achieved_goal'].shape[1] // 2
            achieved_goals = np.concatenate(
                (obs['state_achieved_goal'][:, 0:2],
                 obs['state_achieved_goal'][:, state_size:state_size+2]),
                axis=1
            )
            desired_goals = np.concatenate(
                (obs['state_desired_goal'][:, 0:2],
                obs['state_desired_goal'][:, state_size:state_size+2]),
                axis=1
            )
        else:
            achieved_goals = obs['state_achieved_goal'][:, :2]
            desired_goals = obs['state_desired_goal'][:, :2]
        diff = achieved_goals - desired_goals
        return np.linalg.norm(diff, ord=self.norm_order, axis=1)

    def _compute_xy_successes(self, obs):
        assert not self.two_frames
        achieved_goals = obs['state_achieved_goal'][:, :2]
        desired_goals = obs['state_desired_goal'][:, :2]
        diff = achieved_goals - desired_goals
        return (np.linalg.norm(diff, ord=self.norm_order, axis=1) < 1.50).astype(int)

    def _compute_xy_successes2(self, obs):
        assert not self.two_frames
        achieved_goals = obs['state_achieved_goal'][:, :2]
        desired_goals = obs['state_desired_goal'][:, :2]
        diff = achieved_goals - desired_goals
        return (np.linalg.norm(diff, ord=self.norm_order, axis=1) < 2.50).astype(int)

    def _compute_xy_successes3(self, obs):
        assert not self.two_frames
        achieved_goals = obs['state_achieved_goal'][:, :2]
        desired_goals = obs['state_desired_goal'][:, :2]
        diff = achieved_goals - desired_goals
        return (np.linalg.norm(diff, ord=self.norm_order, axis=1) < 2.75).astype(int)

    def _compute_xy_successes4(self, obs):
        assert not self.two_frames
        achieved_goals = obs['state_achieved_goal'][:, :2]
        desired_goals = obs['state_desired_goal'][:, :2]
        diff = achieved_goals - desired_goals
        return (np.linalg.norm(diff, ord=self.norm_order, axis=1) < 3.00).astype(int)

    def _compute_leg_distances(self, obs):
        if self.use_euler:
            if self.two_frames:
                state_size = obs['state_achieved_goal'].shape[1] // 2
                achieved_goals = np.concatenate(
                    (obs['state_achieved_goal'][:, 9:17],
                     obs['state_achieved_goal'][:, state_size+9:state_size+17]),
                    axis=1
                )
                desired_goals = np.concatenate(
                    (obs['state_desired_goal'][:, 9:17],
                     obs['state_desired_goal'][:, state_size+9:state_size+17]),
                    axis=1
                )
            else:
                achieved_goals = obs['state_achieved_goal'][:, 9:17]
                desired_goals = obs['state_desired_goal'][:, 9:17]
        else:
            if self.two_frames:
                state_size = obs['state_achieved_goal'].shape[1] // 2
                achieved_goals = np.concatenate(
                    (obs['state_achieved_goal'][:, 7:15],
                     obs['state_achieved_goal'][:, state_size+7:state_size+15]),
                    axis=1
                )
                desired_goals = np.concatenate(
                    (obs['state_desired_goal'][:, 7:15],
                     obs['state_desired_goal'][:, state_size+7:state_size+15]),
                    axis=1
                )
            else:
                achieved_goals = obs['state_achieved_goal'][:, 7:15]
                desired_goals = obs['state_desired_goal'][:, 7:15]
        if desired_goals.shape == (1,):
            return -1000
        return np.linalg.norm(
            achieved_goals - desired_goals, ord=self.norm_order, axis=1
        )

    def _compute_quat_distances(self, obs):
        if self.use_euler:
            if self.two_frames:
                state_size = obs['state_achieved_goal'].shape[1] // 2
                achieved_goals = np.concatenate(
                    (self._euler_to_quat(obs['state_achieved_goal'][:, 3:9]),
                     self._euler_to_quat(obs['state_achieved_goal'][:, state_size+3:state_size+9])),
                    axis=1
                )
                desired_goals = np.concatenate(
                    (self._euler_to_quat(obs['state_desired_goal'][:, 3:9]),
                     self._euler_to_quat(obs['state_desired_goal'][:, state_size+3:state_size+9])),
                    axis=1
                )
            else:
                achieved_goals = self._euler_to_quat(obs['state_achieved_goal'][:, 3:9])
                desired_goals = self._euler_to_quat(obs['state_desired_goal'][:, 3:9])
        else:
            if self.two_frames:
                state_size = obs['state_achieved_goal'].shape[1] // 2
                achieved_goals = np.concatenate(
                    (obs['state_achieved_goal'][:, 3:7],
                     obs['state_achieved_goal'][:, state_size+3:state_size+7]),
                    axis=1
                )
                desired_goals = np.concatenate(
                    (obs['state_desired_goal'][:, 3:7],
                     obs['state_desired_goal'][:, state_size+3:state_size+7]),
                    axis=1
                )
            else:
                achieved_goals = obs['state_achieved_goal'][:, 3:7]
                desired_goals = obs['state_desired_goal'][:, 3:7]
        if desired_goals.shape == (1,):
            return -1000
        return np.linalg.norm(
            achieved_goals - desired_goals, ord=self.norm_order, axis=1
        )

    def _compute_euler_distances(self, obs):
        if self.use_euler:
            if self.two_frames:
                state_size = obs['state_achieved_goal'].shape[1] // 2
                achieved_goals = np.concatenate(
                    (obs['state_achieved_goal'][:, 3:9],
                     obs['state_achieved_goal'][:, state_size+3:state_size+9]),
                    axis=1
                )
                desired_goals = np.concatenate(
                    (obs['state_desired_goal'][:, 3:9],
                     obs['state_desired_goal'][:, state_size+3:state_size+9]),
                    axis=1
                )
            else:
                achieved_goals = obs['state_achieved_goal'][:, 3:9]
                desired_goals = obs['state_desired_goal'][:, 3:9]
        else:
            if self.two_frames:
                state_size = obs['state_achieved_goal'].shape[1] // 2
                achieved_goals = np.concatenate(
                    (self._quat_to_euler(obs['state_achieved_goal'][:, 3:7]),
                     self._quat_to_euler(obs['state_achieved_goal'][:, state_size+3:state_size+7])),
                    axis=1
                )
                desired_goals = np.concatenate(
                    (self._quat_to_euler(obs['state_desired_goal'][:, 3:7]),
                     self._quat_to_euler(obs['state_desired_goal'][:, state_size+3:state_size+7])),
                    axis=1
                )
            else:
                achieved_goals = self._quat_to_euler(obs['state_achieved_goal'][:, 3:7])
                desired_goals = self._quat_to_euler(obs['state_desired_goal'][:, 3:7])
        if desired_goals.shape == (1,):
            return -1000
        return np.linalg.norm(
            achieved_goals - desired_goals, ord=self.norm_order, axis=1
        )

    def _compute_qpos_distances(self, obs):
        if self.use_euler:
            if self.two_frames:
                state_size = obs['state_achieved_goal'].shape[1] // 2
                achieved_goals = np.concatenate(
                    (self._epos_to_qpos(obs['state_achieved_goal'][:, 0:self.pos_dim]),
                     self._epos_to_qpos(obs['state_achieved_goal'][:, state_size:state_size + self.pos_dim])),
                    axis=1
                )
                desired_goals = np.concatenate(
                    (self._epos_to_qpos(obs['state_desired_goal'][:, 0:self.pos_dim]),
                     self._epos_to_qpos(obs['state_desired_goal'][:, state_size:state_size + self.pos_dim])),
                    axis=1
                )
            else:
                achieved_goals = self._epos_to_qpos(obs['state_achieved_goal'][:, :self.pos_dim])
                desired_goals = self._epos_to_qpos(obs['state_desired_goal'][:, :self.pos_dim])
        else:
            if self.two_frames:
                state_size = obs['state_achieved_goal'].shape[1] // 2
                achieved_goals = np.concatenate(
                    (obs['state_achieved_goal'][:, 0:self.pos_dim],
                     obs['state_achieved_goal'][:, state_size:state_size+self.pos_dim]),
                    axis=1
                )
                desired_goals = np.concatenate(
                    (obs['state_desired_goal'][:, 0:self.pos_dim],
                     obs['state_desired_goal'][:, state_size:state_size+self.pos_dim]),
                    axis=1
                )
            else:
                achieved_goals = obs['state_achieved_goal'][:, :self.pos_dim]
                desired_goals = obs['state_desired_goal'][:, :self.pos_dim]
        if desired_goals.shape == (1,):
            return -1000
        return np.linalg.norm(
            achieved_goals - desired_goals, ord=self.norm_order, axis=1
        )

    def _compute_epos_distances(self, obs, vectorized=False, sparse=False):
        assert not (sparse and vectorized)
        if not self.use_euler:
            if self.two_frames:
                state_size = obs['state_achieved_goal'].shape[1] // 2
                achieved_goals = np.concatenate(
                    (self._qpos_to_epos(obs['state_achieved_goal'][:, 0:self.pos_dim]),
                     self._qpos_to_epos(obs['state_achieved_goal'][:, state_size:state_size + self.pos_dim])),
                    axis=1
                )
                desired_goals = np.concatenate(
                    (self._qpos_to_epos(obs['state_desired_goal'][:, 0:self.pos_dim]),
                     self._qpos_to_epos(obs['state_desired_goal'][:, state_size:state_size + self.pos_dim])),
                    axis=1
                )
            else:
                achieved_goals = self._qpos_to_epos(obs['state_achieved_goal'][:, :self.pos_dim])
                desired_goals = self._qpos_to_epos(obs['state_desired_goal'][:, :self.pos_dim])
        else:
            if self.two_frames:
                state_size = obs['state_achieved_goal'].shape[1] // 2
                achieved_goals = np.concatenate(
                    (obs['state_achieved_goal'][:, 0:self.pos_dim],
                     obs['state_achieved_goal'][:, state_size:state_size+self.pos_dim]),
                    axis=1
                )
                desired_goals = np.concatenate(
                    (obs['state_desired_goal'][:, 0:self.pos_dim],
                     obs['state_desired_goal'][:, state_size:state_size+self.pos_dim]),
                    axis=1
                )
            else:
                achieved_goals = obs['state_achieved_goal'][:, :self.pos_dim]
                desired_goals = obs['state_desired_goal'][:, :self.pos_dim]
        if desired_goals.shape == (1,):
            return -1000

        d = np.linalg.norm(achieved_goals - desired_goals, ord=self.norm_order, axis=1)
        if sparse:
            return (d > self.sparsity_threshold).astype(np.float32)
        elif not vectorized:
            return d
        else:
            return np.abs(achieved_goals - desired_goals)

    def _compute_epos_weighted_euler_distances(self, obs):
        if not self.use_euler:
            if self.two_frames:
                state_size = obs['state_achieved_goal'].shape[1] // 2
                achieved_goals = np.concatenate(
                    (self._qpos_to_epos(obs['state_achieved_goal'][:, 0:self.pos_dim]),
                     self._qpos_to_epos(obs['state_achieved_goal'][:, state_size:state_size + self.pos_dim])),
                    axis=1
                )
                desired_goals = np.concatenate(
                    (self._qpos_to_epos(obs['state_desired_goal'][:, 0:self.pos_dim]),
                     self._qpos_to_epos(obs['state_desired_goal'][:, state_size:state_size + self.pos_dim])),
                    axis=1
                )
            else:
                achieved_goals = self._qpos_to_epos(obs['state_achieved_goal'][:, :self.pos_dim])
                desired_goals = self._qpos_to_epos(obs['state_desired_goal'][:, :self.pos_dim])
        else:
            if self.two_frames:
                state_size = obs['state_achieved_goal'].shape[1] // 2
                achieved_goals = np.concatenate(
                    (obs['state_achieved_goal'][:, 0:self.pos_dim],
                     obs['state_achieved_goal'][:, state_size:state_size+self.pos_dim]),
                    axis=1
                )
                desired_goals = np.concatenate(
                    (obs['state_desired_goal'][:, 0:self.pos_dim],
                     obs['state_desired_goal'][:, state_size:state_size+self.pos_dim]),
                    axis=1
                )
            else:
                achieved_goals = obs['state_achieved_goal'][:, :self.pos_dim]
                desired_goals = obs['state_desired_goal'][:, :self.pos_dim]
        if desired_goals.shape == (1,):
            return -1000

        scaling_factor = 3.0 #10.0
        achieved_goals = np.copy(achieved_goals)
        desired_goals = np.copy(desired_goals)
        achieved_goals[:,3:9] = achieved_goals[:,3:9] * scaling_factor
        desired_goals[:, 3:9] = desired_goals[:, 3:9] * scaling_factor
        if self.two_frames:
            achieved_goals[:, self.pos_dim+3:self.pos_dim+9] = \
                achieved_goals[:, self.pos_dim+3:self.pos_dim+9] * scaling_factor
            desired_goals[:, self.pos_dim+3:self.pos_dim+9] = \
                desired_goals[:, self.pos_dim+3:self.pos_dim+9] * scaling_factor

        return np.linalg.norm(
            achieved_goals - desired_goals, ord=self.norm_order, axis=1
        )

    def _compute_epos_penalize_flip_distances(self, obs):
        if not self.use_euler:
            if self.two_frames:
                state_size = obs['state_achieved_goal'].shape[1] // 2
                achieved_goals = np.concatenate(
                    (self._qpos_to_epos(obs['state_achieved_goal'][:, 0:self.pos_dim]),
                     self._qpos_to_epos(obs['state_achieved_goal'][:, state_size:state_size + self.pos_dim])),
                    axis=1
                )
                desired_goals = np.concatenate(
                    (self._qpos_to_epos(obs['state_desired_goal'][:, 0:self.pos_dim]),
                     self._qpos_to_epos(obs['state_desired_goal'][:, state_size:state_size + self.pos_dim])),
                    axis=1
                )
            else:
                achieved_goals = self._qpos_to_epos(obs['state_achieved_goal'][:, :self.pos_dim])
                desired_goals = self._qpos_to_epos(obs['state_desired_goal'][:, :self.pos_dim])
        else:
            if self.two_frames:
                state_size = obs['state_achieved_goal'].shape[1] // 2
                achieved_goals = np.concatenate(
                    (obs['state_achieved_goal'][:, 0:self.pos_dim],
                     obs['state_achieved_goal'][:, state_size:state_size+self.pos_dim]),
                    axis=1
                )
                desired_goals = np.concatenate(
                    (obs['state_desired_goal'][:, 0:self.pos_dim],
                     obs['state_desired_goal'][:, state_size:state_size+self.pos_dim]),
                    axis=1
                )
            else:
                achieved_goals = obs['state_achieved_goal'][:, :self.pos_dim]
                desired_goals = obs['state_desired_goal'][:, :self.pos_dim]
        if desired_goals.shape == (1,):
            return -1000

        achieved_goals_flipped = self._check_flipped(achieved_goals[:,3:9])
        desired_goals_flipped = self._check_flipped(desired_goals[:,3:9])
        penalty = 10.0 * np.logical_and(np.logical_not(desired_goals_flipped), achieved_goals_flipped)

        base_distance = np.linalg.norm(
            achieved_goals - desired_goals, ord=self.norm_order, axis=1
        )
        return base_distance + penalty

    def sample_goal(self, mode=None):
        goals = self.sample_goals(1, mode=mode)
        return self.unbatchify_dict(goals, 0)

    def reset_model(self, goal=None):
        self.timesteps_so_far = 0
        if goal is None:
            goal = self.sample_goal()
        self._reset_ant()

        self._set_goal(goal)
        self.sim.forward()
        self._prev_obs = None
        self._cur_obs = None
        self.subgoals = None

        site_xpos = self.sim.data.site_xpos
        start_xpos = np.concatenate((self.sim.data.qpos.flat[:2], np.array([0.5])))
        site_xpos[self.sim.model.site_name2id('start')] = start_xpos
        self.model.site_pos[:] = site_xpos

        return self._get_obs()

    def _reset_ant(self, mode=None):
        # if mode == 'top_left':
        #     qpos = self.init_qpos.copy()
        #     if 'small' in self.model_path:
        #         qpos[:2] = np.random.uniform(
        #             [-2.5, 2.5],
        #             [-2.25, 2.5],
        #             size=(2),
        #         )
        #     elif 'big' in self.model_path:
        #         qpos[:2] = np.random.uniform(
        #             [-4.25, 4.25],
        #             [-3.25, 4.25],
        #             size=(2),
        #         )
        #     else:
        #         raise NotImplementedError
        # elif mode == 'top_right':
        #     qpos = self.init_qpos.copy()
        #     if 'small' in self.model_path:
        #         qpos[:2] = np.random.uniform(
        #             [2.25, 2.5],
        #             [2.5, 2.5],
        #             size=(2),
        #         )
        #     elif 'big' in self.model_path:
        #         qpos[:2] = np.random.uniform(
        #             [3.25, 4.25],
        #             [4.25, 4.25],
        #             size=(2),
        #         )
        #     else:
        #         raise NotImplementedError
        if self.init_xy_mode == 'fixed':
            qpos = self.init_qpos
        elif self.init_xy_mode in ['uniform', 'sample-uniformly-xy-space']:
            qpos = self.init_qpos.copy()
            xy_start = self._sample_uniform_xy(1, mode='reset')[0]
            qpos[:2] = xy_start
        elif self.init_xy_mode == 'uniform_pos_and_rot':
            qpos = self.init_qpos.copy()
            xy_start = self._sample_uniform_xy(1, mode='reset')[0]
            qpos[:2] = xy_start

            rots = np.random.randint(4)
            if rots == 0:
                qpos[3:7] = [1, 0, 0, 0]
            elif rots == 1:
                qpos[3:7] = [0, 0, 0, 1]
            elif rots == 2:
                qpos[3:7] = [0.7071068, 0, 0, 0.7071068]
            elif rots == 3:
                qpos[3:7] = [0.7071068, 0, 0, -0.7071068]

        qvel = np.zeros_like(self.init_qvel)
        self.set_state(qpos, qvel)

    def _set_goal(self, goal):
        if 'state_desired_goal' in goal:
            if self.two_frames:
                state_size = len(goal['state_desired_goal'])
                self._full_state_goal = goal['state_desired_goal'][:state_size//2]
            else:
                self._full_state_goal = goal['state_desired_goal']
        else:
            raise ValueError("C'mon, you gotta give me some goal!")
        self._prev_obs = None
        self._cur_obs = None

        site_xpos = self.sim.data.site_xpos
        goal_xpos = np.concatenate((self._full_state_goal[:2], np.array([0.75])))
        site_xpos[self.sim.model.site_name2id('goal')] = goal_xpos
        self.model.site_pos[:] = site_xpos

    def set_to_goal(self, goal):
        state_goal = goal['state_desired_goal']
        if self.use_euler:
            qpos = self._epos_to_qpos(state_goal[:self.pos_dim])
        else:
            qpos = state_goal[:self.pos_dim]

        if self.vel_in_state:
            qvel = state_goal[self.pos_dim:self.state_dim]
        else:
            qvel = np.zeros(14)
        self.set_state(qpos, qvel)
        self._prev_obs = None
        self._cur_obs = None

    def get_env_state(self):
        joint_state = self.sim.get_state()
        goal = self._full_state_goal
        state = joint_state, goal, self._cur_obs, self._prev_obs
        return copy.deepcopy(state)

    def set_env_state(self, state):
        joint_state, goal, cur_obs, prev_obs = state
        self.sim.set_state(joint_state)
        self.sim.forward()
        self._full_state_goal = goal
        self._cur_obs = cur_obs
        self._prev_obs = prev_obs

    def update_subgoals(self, subgoals, **kwargs):
        self.subgoals = subgoals

    def get_image_plt(
            self,
            vals=None,
            vmin=None, vmax=None,
            extent=None,
            small_markers=False,
            draw_walls=True, draw_state=True, draw_goal=False, draw_subgoals=False,
            imsize=400 #400
    ):
        if self.model_path in [
            'classic_mujoco/ant_maze2_gear30_small_dt3.xml',
            'classic_mujoco/ant_gear30_dt3_u_small.xml',
        ]:
            extent = [-3.5, 3.5, -3.5, 3.5]
        elif self.model_path in [
            'classic_mujoco/ant_gear30_dt3_u_med.xml',
            'classic_mujoco/ant_gear15_dt3_u_med.xml',
            'classic_mujoco/ant_gear10_dt3_u_med.xml',
            'classic_mujoco/ant_gear30_dt2_u_med.xml',
            'classic_mujoco/ant_gear15_dt2_u_med.xml',
            'classic_mujoco/ant_gear10_dt2_u_med.xml',
            'classic_mujoco/ant_gear30_u_med.xml',
        ]:
            extent = [-4.5, 4.5, -4.5, 4.5]
        elif self.model_path in [
            'classic_mujoco/ant_fb_gear30_med_dt3.xml',
            'classic_mujoco/ant_gear30_dt3_fb_med.xml',
            'classic_mujoco/ant_gear15_dt3_fb_med.xml',
            'classic_mujoco/ant_gear10_dt3_fb_med.xml',
        ]:
            extent = [-6.0, 6.0, -6.0, 6.0]
        elif self.model_path in [
            'classic_mujoco/ant_gear10_dt3_u_long.xml',
            'classic_mujoco/ant_gear15_dt3_u_long.xml',
        ]:
            extent = [-3.75, 3.75, -9.0, 9.0]
        elif self.model_path in [
            'classic_mujoco/ant_gear10_dt3_maze_med.xml',
        ]:
            extent = [-8.0, 8.0, -8.0, 8.0]
        elif self.model_path in [
            'classic_mujoco/ant_gear10_dt3_fg_med.xml',
        ]:
            extent = [-8.0, 8.0, -8.0, 8.0]
        else:
            extent = [-5.5, 5.5, -5.5, 5.5]

        fig, ax = plt.subplots()
        ax.set_ylim(extent[2:4])
        ax.set_xlim(extent[0:2])
        # ax.set_xlim([2.75, -2.75])
        # ax.set_ylim(ax.get_ylim()[::-1])
        DPI = fig.get_dpi()
        fig.set_size_inches(imsize / float(DPI), imsize / float(DPI))

        marker_factor = 0.60
        if small_markers:
            marker_factor = 0.10

        ob = self._get_obs()

        if draw_state:
            if small_markers:
                color = 'cyan'
            else:
                color = 'blue'

            if self._check_flipped(ob['state_observation'][3:9][None])[0]:
                color = 'purple'

            ball = plt.Circle(ob['state_observation'][:2], 0.50 * marker_factor, color=color)
            ax.add_artist(ball)
        if draw_goal:
            goal = plt.Circle(ob['state_desired_goal'][:2], 0.50 * marker_factor, color='green')
            ax.add_artist(goal)
        if draw_subgoals:
            if self.subgoals is not None:
                for i in range(len(self.subgoals)):
                    subgoal = plt.Circle(self.subgoals[i][:2], (0.50 - 0.1) * marker_factor, color='red')
                    ax.add_artist(subgoal)
        if draw_walls:# and hasattr(self, "walls:"):
            for wall in self.walls:
                # ax.vlines(x=wall.min_x, ymin=wall.min_y, ymax=wall.max_y)
                # ax.hlines(y=wall.min_y, xmin=wall.min_x, xmax=wall.max_x)
                # ax.vlines(x=wall.max_x, ymin=wall.min_y, ymax=wall.max_y)
                # ax.hlines(y=wall.max_y, xmin=wall.min_x, xmax=wall.max_x)
                ax.vlines(x=wall.endpoint1[0], ymin=wall.endpoint2[1], ymax=wall.endpoint1[1])
                ax.hlines(y=wall.endpoint2[1], xmin=wall.endpoint3[0], xmax=wall.endpoint2[0])
                ax.vlines(x=wall.endpoint3[0], ymin=wall.endpoint3[1], ymax=wall.endpoint4[1])
                ax.hlines(y=wall.endpoint4[1], xmin=wall.endpoint4[0], xmax=wall.endpoint1[0])

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        fig.subplots_adjust(bottom=0)
        fig.subplots_adjust(top=1)
        fig.subplots_adjust(right=1)
        fig.subplots_adjust(left=0)
        ax.axis('off')

        # if vals is not None:
        if type(vals) is np.ndarray:
            ax.imshow(
                vals,
                extent=extent,
                cmap=plt.get_cmap('plasma'),
                interpolation='nearest',
                vmax=vmax,
                vmin=vmin,
                origin='bottom',  # <-- Important! By default top left is (0, 0)
            )

        return self.plt_to_numpy(fig)

    def plt_to_numpy(self, fig):
        fig.canvas.draw()
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()
        return data

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
