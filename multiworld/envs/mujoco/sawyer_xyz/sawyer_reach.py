from collections import OrderedDict
import numpy as np
from gym.spaces import Box, Dict

from multiworld.envs.env_util import get_stat_in_paths, \
    create_stats_ordered_dict, get_asset_full_path
from multiworld.core.multitask_env import MultitaskEnv
from multiworld.envs.mujoco.sawyer_xyz.base import SawyerXYZEnv

import matplotlib.pyplot as plt


class SawyerReachXYZEnv(SawyerXYZEnv, MultitaskEnv):
    def __init__(
            self,
            reward_type='hand_distance',
            norm_order=1,
            indicator_threshold=0.06,

            fix_reset=True,
            fixed_reset=(0, 0.5, 0.02),
            fix_goal=False,
            fixed_goal=(0.15, 0.6, 0.3),
            hide_goal_markers=False,

            **kwargs
    ):
        self.quick_init(locals())
        MultitaskEnv.__init__(self)
        SawyerXYZEnv.__init__(self, model_name=self.model_name, **kwargs)

        self.reward_type = reward_type
        self.norm_order = norm_order
        self.indicator_threshold = indicator_threshold

        self.fix_reset = fix_reset
        self.fixed_reset = np.array(fixed_reset)
        self.fix_goal = fix_goal
        self.fixed_goal = np.array(fixed_goal)
        self._state_goal = None
        self.hide_goal_markers = hide_goal_markers
        self.action_space = Box(np.array([-1, -1, -1]), np.array([1, 1, 1]), dtype=np.float32)
        self.hand_space = Box(self.hand_low, self.hand_high, dtype=np.float32)
        self.observation_space = Dict([
            ('observation', self.hand_space),
            ('desired_goal', self.hand_space),
            ('achieved_goal', self.hand_space),
            ('state_observation', self.hand_space),
            ('state_desired_goal', self.hand_space),
            ('state_achieved_goal', self.hand_space),
            ('proprio_observation', self.hand_space),
            ('proprio_desired_goal', self.hand_space),
            ('proprio_achieved_goal', self.hand_space),
        ])
        self.reset()

    def step(self, action):
        self.set_xyz_action(action)
        # keep gripper closed
        self.do_simulation(np.array([1]))
        # The marker seems to get reset every time you do a simulation
        self._set_goal_marker(self._state_goal)
        ob = self._get_obs()
        reward = self.compute_reward(action, ob)
        info = self._get_info()
        done = False
        return ob, reward, done, info

    def _get_obs(self):
        flat_obs = self.get_endeff_pos()
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

    def _get_info(self):
        hand_diff = self._state_goal - self.get_endeff_pos()
        hand_distance = np.linalg.norm(hand_diff, ord=self.norm_order)
        hand_distance_l1 = np.linalg.norm(hand_diff, ord=1)
        hand_distance_l2 = np.linalg.norm(hand_diff, ord=2)
        return dict(
            hand_distance=hand_distance,
            hand_distance_l1=hand_distance_l1,
            hand_distance_l2=hand_distance_l2,
            hand_success=float(hand_distance < self.indicator_threshold),
        )

    def _set_goal_marker(self, goal):
        """
        This should be use ONLY for visualization. Use self._state_goal for
        logging, learning, etc.
        """
        self.data.site_xpos[self.model.site_name2id('hand-goal-site')] = (
            goal
        )
        if self.hide_goal_markers:
            self.data.site_xpos[self.model.site_name2id('hand-goal-site'), 2] = (
                -1000
            )

    @property
    def model_name(self):
        return get_asset_full_path('sawyer_xyz/sawyer_reach.xml')

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.lookat[0] = 0
        self.viewer.cam.lookat[1] = 1.0
        self.viewer.cam.lookat[2] = 0.5
        self.viewer.cam.distance = 0.3
        self.viewer.cam.elevation = -45
        self.viewer.cam.azimuth = 270
        self.viewer.cam.trackbodyid = -1

    def reset_model(self):
        velocities = self.data.qvel.copy()
        angles = self.data.qpos.copy()
        angles[:7] = [1.7244448, -0.92036369,  0.10234232,  2.11178144,  2.97668632, -0.38664629, 0.54065733]
        self.set_state(angles.flatten(), velocities.flatten())
        self._reset_hand()
        self.set_goal(self.sample_goal())
        self.sim.forward()
        return self._get_obs()

    def _reset_hand(self):
        if self.fix_reset:
            new_mocap_pos = self.fixed_reset
        else:
            new_mocap_pos = np.random.uniform(self.hand_space.low, self.hand_space.high)
        for _ in range(10):
            self.data.set_mocap_pos('mocap', new_mocap_pos)
            self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
            self.do_simulation(None, self.frame_skip)

    """
    Multitask functions
    """
    def get_goal(self):
        return {
            'desired_goal': self._state_goal,
            'state_desired_goal': self._state_goal,
        }

    def set_goal(self, goal):
        self._state_goal = goal['state_desired_goal']
        self._set_goal_marker(self._state_goal)

    def set_to_goal(self, goal):
        state_goal = goal['state_desired_goal']
        for _ in range(30):
            self.data.set_mocap_pos('mocap', state_goal)
            self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
            # keep gripper closed
            self.do_simulation(np.array([1]))

    def sample_goals(self, batch_size):
        if self.fix_goal:
            goals = np.repeat(
                self.fixed_goal.copy()[None],
                batch_size,
                0
            )
        else:
            goals = np.random.uniform(
                self.hand_space.low,
                self.hand_space.high,
                size=(batch_size, self.hand_space.low.size),
            )
        return {
            'desired_goal': goals,
            'state_desired_goal': goals,
        }

    def compute_rewards(self, actions, obs):
        achieved_goals = obs['state_achieved_goal']
        desired_goals = obs['state_desired_goal']
        hand_pos = achieved_goals
        goals = desired_goals
        hand_diff = hand_pos - goals
        if self.reward_type == 'hand_distance':
            r = -np.linalg.norm(hand_diff, ord=self.norm_order, axis=1)
        elif self.reward_type == 'vectorized_hand_distance':
            r = -np.abs(hand_diff)
        elif self.reward_type == 'hand_success':
            r = -(np.linalg.norm(hand_diff, ord=self.norm_order, axis=1)
                  > self.indicator_threshold).astype(float)
        else:
            raise NotImplementedError("Invalid/no reward type.")
        return r

    def get_diagnostics(self, paths, prefix=''):
        statistics = OrderedDict()
        for stat_name in [
            'hand_distance',
            'hand_distance_l1',
            'hand_distance_l2',
            'hand_success',
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

    def get_env_state(self):
        base_state = super().get_env_state()
        goal = self._state_goal.copy()
        return base_state, goal

    def set_env_state(self, state):
        base_state, goal = state
        super().set_env_state(base_state)
        self._state_goal = goal
        self._set_goal_marker(goal)


class SawyerReachXYEnv(SawyerReachXYZEnv):
    def __init__(self, *args,
                 fixed_goal=(0.15, 0.6),
                 hand_z_position=0.055, **kwargs):
        self.quick_init(locals())
        SawyerReachXYZEnv.__init__(
            self,
            *args,
            fixed_goal=(fixed_goal[0], fixed_goal[1], hand_z_position),
            **kwargs
        )
        self.hand_z_position = hand_z_position
        self.action_space = Box(np.array([-1, -1]), np.array([1, 1]), dtype=np.float32)
        self.hand_space = Box(
            np.hstack((self.hand_space.low[:2], self.hand_z_position)),
            np.hstack((self.hand_space.high[:2], self.hand_z_position)),
            dtype=np.float32
        )
        self.observation_space = Dict([
            ('observation', self.hand_space),
            ('desired_goal', self.hand_space),
            ('achieved_goal', self.hand_space),
            ('state_observation', self.hand_space),
            ('state_desired_goal', self.hand_space),
            ('state_achieved_goal', self.hand_space),
            ('proprio_observation', self.hand_space),
            ('proprio_desired_goal', self.hand_space),
            ('proprio_achieved_goal', self.hand_space),
        ])

        x_bounds = np.array([self.hand_space.low[0] - 0.03, self.hand_space.high[0] + 0.03])
        y_bounds = np.array([self.hand_space.low[1] - 0.03, self.hand_space.high[1] + 0.03])
        self.vis_bounds = np.concatenate((x_bounds, y_bounds))

    def step(self, action):
        delta_z = self.hand_z_position - self.data.mocap_pos[0, 2]
        action = np.hstack((action, delta_z))
        return super().step(action)

    def get_states_sweep(self, nx, ny):
        x = np.linspace(self.vis_bounds[0], self.vis_bounds[1], nx)
        y = np.linspace(self.vis_bounds[2], self.vis_bounds[3], ny)
        xv, yv = np.meshgrid(x, y)
        zv = self.hand_z_position * np.ones(xv.shape)
        states = np.stack((xv, yv, zv), axis=2).reshape((nx*ny, -1))
        return states

    def get_image_plt(self,
                      vals,
                      vmin=None, vmax=None,
                      extent=None,
                      imsize=84,
                      small_markers=False,
                      draw_walls=True, draw_state=True, draw_goal=False):
        if extent is None:
            extent = self.vis_bounds

        # if vmin is None and vmax is None:
        #     nx, ny = vals.shape[0], vals.shape[1]
        #     vals_within_boundary = vals[int(nx/6):-int(nx/6), int(ny/6):-int(ny/6)]
        #     vmin, vmax = np.min(vals_within_boundary), np.max(vals_within_boundary)

        fig, ax = plt.subplots()
        ax.set_ylim(extent[2:4])
        ax.set_xlim(extent[0:2])
        ax.set_ylim(ax.get_ylim()[::-1])
        ax.set_xlim(ax.get_xlim()[::-1])
        DPI = fig.get_dpi()
        fig.set_size_inches(imsize / float(DPI), imsize / float(DPI))

        marker_factor = 0.60
        if small_markers:
            marker_factor = 0.10

        hand_low, hand_high = self.hand_space.low, self.hand_space.high
        ax.vlines(x=hand_low[0], ymin=hand_low[1], ymax=hand_high[1], linestyles='dotted')
        ax.hlines(y=hand_low[1], xmin=hand_low[0], xmax=hand_high[0], linestyles='dotted')
        ax.vlines(x=hand_high[0], ymin=hand_low[1], ymax=hand_high[1], linestyles='dotted')
        ax.hlines(y=hand_high[1], xmin=hand_low[0], xmax=hand_high[0], linestyles='dotted')

        if draw_state:
            if small_markers:
                color = 'cyan'
            else:
                color = 'blue'
            ball = plt.Circle(self.get_endeff_pos()[:2], 0.03 * marker_factor, color=color)
            ax.add_artist(ball)
        if draw_goal:
            goal = plt.Circle(self._state_goal[:2], 0.03 * marker_factor, color='green')
            ax.add_artist(goal)

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        fig.subplots_adjust(bottom=0)
        fig.subplots_adjust(top=1)
        fig.subplots_adjust(right=1)
        fig.subplots_adjust(left=0)

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
