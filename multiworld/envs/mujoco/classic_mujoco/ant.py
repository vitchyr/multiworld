import abc
import copy
import numpy as np
from gym.spaces import Box, Dict

from multiworld.core.serializable import Serializable
from multiworld.envs.mujoco.mujoco_env import MujocoEnv
from multiworld.core.multitask_env import MultitaskEnv
from multiworld.envs.env_util import get_asset_full_path
import os.path as osp
from railrl.misc.asset_loader import load_local_or_remote_file

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from collections import OrderedDict

from multiworld.envs.env_util import get_stat_in_paths, create_stats_ordered_dict

PRESET1 = np.array([
    [-3, 0],
    [0, -3],
])
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
            init_qpos=None,
            fixed_goal=None, # deprecated feature
            init_xy_mode='fixed',
            terminate_when_unhealthy=False,
            healthy_z_range=(0.2, 0.9),
            health_reward=10,
            goal_sampling_strategy='uniform',
            presampled_goal_paths='',
            fixed_goal_qpos=None,
            test_mode_case_num=None,
            *args,
            **kwargs):
        assert init_xy_mode in {
            'fixed',
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
        self.fixed_goal_qpos = fixed_goal_qpos
        self.init_xy_mode = init_xy_mode
        self.terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_reward = health_reward
        self._healthy_z_range = healthy_z_range

        self.model_path = model_path
        assert goal_sampling_strategy in {
            'uniform',
            'uniform_pos_and_rot',
            'preset1',
            'presampled'
        }
        self.goal_sampling_strategy = goal_sampling_strategy
        if self.goal_sampling_strategy == 'presampled':
            assert presampled_goal_paths is not None
            # if not osp.exists(presampled_goal_paths):
            #     presampled_goal_paths = get_asset_full_path(
            #         presampled_goal_paths
            #     )
            # self.presampled_goals = np.load(presampled_goal_paths)
            self.presampled_goals = load_local_or_remote_file(presampled_goal_paths)
        else:
            self.presampled_goals = None

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
        self._xy_goal = None
        self._qpos_goal = None
        self._prev_obs = None
        self._cur_obs = None
        self.subgoals = None
        self.test_mode_case_num = test_mode_case_num
        self.reset()

    def step(self, action):
        self._prev_obs = self._cur_obs
        self.do_simulation(np.array(action), self.frame_skip)
        ob = self._get_obs()
        reward = self.compute_reward(action, ob)
        info = {}
        if self._full_state_goal is not None:
            info['full-state-distance'] = self._compute_state_distances(
                self.numpy_batchify_dict(ob)
            )
        if self._qpos_goal is not None:
            info['qpos-distance'] = self._compute_qpos_distances(
                self.numpy_batchify_dict(ob)
            )
        if self._xy_goal is not None:
            info['xy-distance'] = self._compute_xy_distances(
                self.numpy_batchify_dict(ob)
            )
        if self.terminate_when_unhealthy:
            done = not self.is_healthy
            reward += self._healthy_reward
        else:
            done = False

        info['is_not_healthy'] = int(not self.is_healthy)

        if len(self.init_qpos) > 15 and self.viewer is not None:
            qpos = self.sim.data.qpos
            qpos[15:] = self._full_state_goal[:15]
            qvel = self.sim.data.qvel
            self.set_state(qpos, qvel)
        return ob, reward, done, info

    def get_diagnostics(self, paths, prefix=''):
        statistics = OrderedDict()
        for stat_name in [
            'full-state-distance',
            'qpos-distance',
            'xy-distance',
            'is_not_healthy',
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

    def sample_goals(self, batch_size, mode=None):
        if mode == 'top_left':
            qpos = self.init_qpos.copy().reshape(1, -1)
            qpos = np.tile(qpos, (batch_size, 1))
            qpos[:,:2] = np.random.uniform(
                [-2.5, 2.5],
                [-2.25, 2.5],
                size=(batch_size, 2),
            )

            if self.vel_in_state:
                qvel = np.zeros((batch_size, 14))
                state_goals = np.concatenate((qpos, qvel), axis=1)
        elif mode == 'top_right':
            qpos = self.init_qpos.copy().reshape(1, -1)
            qpos = np.tile(qpos, (batch_size, 1))
            qpos[:,:2] = np.random.uniform(
                [2.25, 2.5],
                [2.5, 2.5],
                size=(batch_size, 2),
            )

            if self.vel_in_state:
                qvel = np.zeros((batch_size, 14))
                state_goals = np.concatenate((qpos, qvel), axis=1)
        elif self.fixed_goal_qpos is not None:
            fixed_goal = self.fixed_goal_qpos
            if self.vel_in_state:
                fixed_goal = np.concatenate((fixed_goal, np.zeros(14)))
            state_goals = np.tile(fixed_goal, (batch_size, 1))
        elif self.goal_sampling_strategy == 'uniform':
            qpos = self.init_qpos.copy().reshape(1, -1)
            qpos = np.tile(qpos, (batch_size, 1))
            qpos[:,:2] = self._sample_uniform_xy(batch_size)

            if self.vel_in_state:
                qvel = np.zeros((batch_size, 14))
                state_goals = np.concatenate((qpos, qvel), axis=1)
        elif self.goal_sampling_strategy == 'uniform_pos_and_rot':
            qpos = self.init_qpos.copy().reshape(1, -1)
            qpos = np.tile(qpos, (batch_size, 1))
            qpos[:,:2] = self._sample_uniform_xy(batch_size)

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

            if self.vel_in_state:
                qvel = np.zeros((batch_size, 14))
                state_goals = np.concatenate((qpos, qvel), axis=1)
        elif self.goal_sampling_strategy == 'preset1':
            assert self.goal_is_xy and self.vel_in_state
            raise NotImplementedError()
            # xy_goals = PRESET1[
            #     np.random.randint(PRESET1.shape[0], size=batch_size), :
            # ]
        elif self.goal_sampling_strategy == 'presampled':
            idxs = np.random.randint(
                self.presampled_goals.shape[0], size=batch_size,
            )
            state_goals = self.presampled_goals[idxs, :]
            if not self.vel_in_state:
                state_goals = state_goals[:, :15]
        else:
            raise NotImplementedError(self.goal_sampling_strategy)

        if self.two_frames:
            state_goals = np.concatenate((state_goals, state_goals), axis=1)

        goals_dict = {
            'desired_goal': state_goals.copy(),
            'state_desired_goal': state_goals.copy(),
        }

        return goals_dict

    def _sample_uniform_xy(self, batch_size):
        goals = np.random.uniform(
            self.goal_space.low[:2],
            self.goal_space.high[:2],
            size=(batch_size, 2),
        )
        return goals

    def compute_rewards(self, actions, obs, prev_obs=None):
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

    def _compute_state_distances(self, obs):
        achieved_goals = obs['state_achieved_goal']
        desired_goals = obs['state_desired_goal']
        if desired_goals.shape == (1,):
            return -1000
        ant_pos = achieved_goals
        goals = desired_goals
        diff = ant_pos - goals
        return np.linalg.norm(diff, ord=self.norm_order, axis=1)

    def _compute_qpos_distances(self, obs):
        if self.two_frames:
            state_size = obs['state_achieved_goal'].shape[1] // 2
            achieved_goals = np.concatenate(
                (obs['state_achieved_goal'][:, 0:15],
                 obs['state_achieved_goal'][:, state_size:state_size+15]),
                axis=1
            )
            desired_goals = np.concatenate(
                (obs['state_desired_goal'][:, 0:15],
                 obs['state_desired_goal'][:, state_size:state_size+15]),
                axis=1
            )
        else:
            achieved_goals = obs['state_achieved_goal'][:, :15]
            desired_goals = obs['state_desired_goal'][:, :15]
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

    def sample_goal(self, mode=None):
        goals = self.sample_goals(1, mode=mode)
        return self.unbatchify_dict(goals, 0)

    def reset_model(self, goal=None):
        if self.test_mode_case_num == 1:
            if goal is None:
                goal = self.sample_goal(mode='top_right')
            self._reset_ant(mode='top_left')
        elif self.test_mode_case_num == 2:
            if goal is None:
                goal = self.sample_goal(mode='top_left')
            self._reset_ant(mode='top_right')
        elif self.test_mode_case_num == 3:
            mode = np.random.randint(2)
            if mode == 0:
                if goal is None:
                    goal = self.sample_goal(mode='top_left')
                self._reset_ant(mode='top_right')
            elif mode == 1:
                if goal is None:
                    goal = self.sample_goal(mode='top_right')
                self._reset_ant(mode='top_left')
        else:
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
        if mode == 'top_left':
            qpos = self.init_qpos.copy()
            qpos[:2] = np.random.uniform(
                [-2.5, 2.5],
                [-2.25, 2.5],
                size=(2),
            )
        elif mode == 'top_right':
            qpos = self.init_qpos.copy()
            qpos[:2] = np.random.uniform(
                [2.25, 2.5],
                [2.5, 2.5],
                size=(2),
            )
        elif self.init_xy_mode == 'fixed':
            qpos = self.init_qpos
        elif self.init_xy_mode == 'sample-uniformly-xy-space':
            qpos = self.init_qpos.copy()
            xy_start = self._sample_uniform_xy(1)[0]
            qpos[:2] = xy_start
        qvel = np.zeros_like(self.init_qvel)
        self.set_state(qpos, qvel)

    def _set_goal(self, goal):
        if 'state_desired_goal' in goal:
            if self.two_frames:
                state_size = len(goal['state_desired_goal'])
                self._full_state_goal = goal['state_desired_goal'][:state_size//2]
            else:
                self._full_state_goal = goal['state_desired_goal']
            self._qpos_goal = self._full_state_goal[:15]
            self._xy_goal = self._qpos_goal[:2]
            # if 'qpos_desired_goal' in goal:
            #     assert (self._qpos_goal == goal['qpos_desired_goal'][:15]).all()
            # if 'xy_desired_goal' in goal:
            #     assert (self._xy_goal == goal['xy_desired_goal'][:2]).all()
        # elif 'qpos_desired_goal' in goal:
        #     raise NotImplementedError
        #     self._full_state_goal = None
        #     self._qpos_goal = goal['qpos_desired_goal']
        #     self._xy_goal = self._qpos_goal[:2]
        #     # if 'xy_desired_goal' in goal:
        #     #     assert (self._xy_goal == goal['xy_desired_goal']).all()
        # elif 'xy_desired_goal' in goal:
        #     raise NotImplementedError
        #     self._full_state_goal = None
        #     self._qpos_goal = None
        #     self._xy_goal = goal['xy_desired_goal']
        else:
            raise ValueError("C'mon, you gotta give me some goal!")
        assert self._xy_goal is not None
        self._prev_obs = None
        self._cur_obs = None
        if len(self.init_qpos) > 15 and self._qpos_goal is not None:
            qpos = self.init_qpos
            qpos[15:] = self._qpos_goal
            qvel = self.sim.data.qvel
            self.set_state(qpos, qvel)
        else:
            site_xpos = self.sim.data.site_xpos
            goal_xpos = np.concatenate((self._xy_goal[:2], np.array([0.75])))
            site_xpos[self.sim.model.site_name2id('goal')] = goal_xpos
            self.model.site_pos[:] = site_xpos

    def set_to_goal(self, goal):
        state_goal = goal['state_desired_goal']
        qpos = state_goal[0:15]
        if self.vel_in_state:
            qvel = state_goal[15:29]
        else:
            qvel = np.zeros(14)
        self.set_state(qpos, qvel)
        self._prev_obs = None
        self._cur_obs = None
        # qpos = list(self.sim.data.qpos.flat)[:15]
        # flat_obs = qpos
        # if self.vel_in_state:
        #     flat_obs = flat_obs + list(self.sim.data.qvel.flat[:14])
        # flat_obs = np.array(flat_obs)
        # print(flat_obs - state_goal)

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
            extent=[-3.5, 3.5, -3.5, 3.5],
            small_markers=False,
            draw_walls=True, draw_state=True, draw_goal=False, draw_subgoals=False,
            imsize=84
    ):
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
        # if draw_walls:
        #     for wall in self.walls:
        #         # ax.vlines(x=wall.min_x, ymin=wall.min_y, ymax=wall.max_y)
        #         # ax.hlines(y=wall.min_y, xmin=wall.min_x, xmax=wall.max_x)
        #         # ax.vlines(x=wall.max_x, ymin=wall.min_y, ymax=wall.max_y)
        #         # ax.hlines(y=wall.max_y, xmin=wall.min_x, xmax=wall.max_x)
        #         ax.vlines(x=wall.endpoint1[0], ymin=wall.endpoint2[1], ymax=wall.endpoint1[1])
        #         ax.hlines(y=wall.endpoint2[1], xmin=wall.endpoint3[0], xmax=wall.endpoint2[0])
        #         ax.vlines(x=wall.endpoint3[0], ymin=wall.endpoint3[1], ymax=wall.endpoint4[1])
        #         ax.hlines(y=wall.endpoint4[1], xmin=wall.endpoint4[0], xmax=wall.endpoint1[0])

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        fig.subplots_adjust(bottom=0)
        fig.subplots_adjust(top=1)
        fig.subplots_adjust(right=1)
        fig.subplots_adjust(left=0)
        ax.axis('off')

        # if vals is not None:
        #     ax.imshow(
        #         vals,
        #         extent=extent,
        #         cmap=plt.get_cmap('plasma'),
        #         interpolation='nearest',
        #         vmax=vmax,
        #         vmin=vmin,
        #         origin='bottom',  # <-- Important! By default top left is (0, 0)
        #     )

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
