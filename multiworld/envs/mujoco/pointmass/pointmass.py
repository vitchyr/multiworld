import abc
import copy
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

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

class PointmassEnv(MujocoEnv, Serializable, MultitaskEnv):
    def __init__(
            self,
            reward_type='dense',
            norm_order=2,
            action_scale=0.15,
            frame_skip=100,
            ball_low=(-4.0, -4.0),
            ball_high=(4.0, 4.0),
            goal_low=(-4.0, -4.0),
            goal_high=(4.0, 4.0),
            model_path='pointmass_u_wall_big.xml',
            reset_low=None,
            reset_high=None,
            v_func_heatmap_bounds=(-2.0, 0.0),
            *args,
            **kwargs
    ):
        self.quick_init(locals())
        MultitaskEnv.__init__(self)
        MujocoEnv.__init__(self,
                           model_path=get_asset_full_path('pointmass/' + model_path),
                           frame_skip=frame_skip,
                           **kwargs)

        self.action_space = Box(np.array([-1, -1]), np.array([1, 1]), dtype=np.float32)
        self.action_scale = action_scale
        self.ball_radius = 0.50
        self.walls = [
            Wall(0, 1.0, 2.5, 0.5, self.ball_radius),
            Wall(2.0, -0.5, 0.5, 2.0, self.ball_radius),
            Wall(-2.0, -0.5, 0.5, 2.0, self.ball_radius),
        ]

        self.reward_type = reward_type
        self.norm_order = norm_order

        self.ball_low, self.ball_high = np.array(ball_low), np.array(ball_high)
        self.goal_low, self.goal_high = np.array(goal_low), np.array(goal_high)

        if reset_low is None:
            self.reset_low = np.array(ball_low)
        else:
            self.reset_low = np.array(reset_low)
        if reset_high is None:
            self.reset_high = np.array(ball_high)
        else:
            self.reset_high = np.array(reset_high)

        obs_space_low = np.copy(self.ball_low)
        obs_space_high = np.copy(self.ball_high)
        goal_space_low = np.copy(self.goal_low)
        goal_space_high = np.copy(self.goal_high)

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

        self.v_func_heatmap_bounds = v_func_heatmap_bounds

        self._state_goal = None
        self.reset()

    def step(self, velocities):
        velocities = np.clip(velocities, a_min=-1, a_max=1) * self.action_scale
        ob = self._get_obs()
        action = velocities

        self.do_simulation(action, self.frame_skip)

        state, goal = ob['state_observation'], ob['state_desired_goal']
        distance_to_target = np.linalg.norm(state - goal)
        is_success = distance_to_target < 1.0
        is_success_2 = distance_to_target < 1.5
        is_success_3 = distance_to_target < 1.75

        ob = self._get_obs()
        reward = self.compute_reward(action, ob)
        info = {
            'distance_to_target': distance_to_target,
            'is_success': is_success,
            'is_success_2': is_success_2,
            'is_success_3': is_success_3,
            'velocity': velocities,
            'speed': np.linalg.norm(velocities),
        }
        done = False
        return ob, reward, done, info

    def _get_obs(self):
        qpos = list(self.sim.data.qpos.flat)
        flat_obs = np.array(qpos)

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

        return self._cur_obs

    def get_goal(self):
        return {
            'desired_goal': self._state_goal.copy(),
            'state_desired_goal': self._state_goal.copy(),
        }

    def sample_goals(self, batch_size):
        goals = np.random.uniform(
            self.goal_space.low,
            self.goal_space.high,
            size=(batch_size, self.goal_space.low.size),
        )

        collisions = self._positions_inside_wall(goals[:,:2])
        collision_idxs = np.where(collisions)[0]
        while len(collision_idxs) > 0:
            goals[collision_idxs,:2] = np.random.uniform(
                self.goal_space.low[:2],
                self.goal_space.high[:2],
                size=(len(collision_idxs), 2)
            )
            collisions = self._positions_inside_wall(goals[:, :2])
            collision_idxs = np.where(collisions)[0]

        return {
            'desired_goal': goals,
            'state_desired_goal': goals,
        }

    def _positions_inside_wall(self, positions):
        inside_wall = False
        for wall in self.walls:
            inside_wall = inside_wall | wall.contains_points(positions)
        return inside_wall

    def _position_inside_wall(self, pos):
        for wall in self.walls:
            if wall.contains_point(pos):
                return True
        return False

    def compute_rewards(self, actions, obs, prev_obs=None):
        achieved_goals = obs['state_achieved_goal']
        desired_goals = obs['state_desired_goal']
        diff = achieved_goals - desired_goals
        if self.reward_type == 'vectorized_dense':
            r = -np.abs(diff)
        elif self.reward_type == 'dense':
            r = -np.linalg.norm(diff, ord=self.norm_order, axis=1)
        else:
            raise NotImplementedError("Invalid/no reward type.")
        return r

    def reset_model(self):
        self._reset_ball()
        self.set_goal(self.sample_goal())
        self.sim.forward()
        return self._get_obs()

    def _reset_ball(self):
        qvel = np.zeros(2)
        pos_2d = np.random.uniform(self.reset_low, self.reset_high)
        while self._position_inside_wall(pos_2d):
            pos_2d = np.random.uniform(self.reset_low, self.reset_high)
        qpos = pos_2d
        self.set_state(qpos, qvel)

    def set_goal(self, goal):
        self._state_goal = goal['state_desired_goal']

    def set_to_goal(self, goal):
        qpos = goal['state_desired_goal']
        qvel = np.zeros(2)
        self.set_state(qpos, qvel)

    def get_env_state(self):
        joint_state = self.sim.get_state()
        state = joint_state, self._state_goal
        return copy.deepcopy(state)

    def set_env_state(self, state):
        state, goal = state
        self.sim.set_state(state)
        self.sim.forward()
        self._state_goal = goal

    def get_diagnostics(self, paths, prefix=''):
        statistics = OrderedDict()
        list_of_stat_names = [
            'distance_to_target',
            'is_success',
            'is_success_2',
            'is_success_3',
            'velocity',
            'speed',
        ]
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
        # self.viewer.cam.distance = 12.5
        # self.viewer.cam.elevation = -90
        # self.viewer.cam.azimuth = 270

        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.lookat[0] = 0.0
        self.viewer.cam.lookat[1] = 0.75
        self.viewer.cam.lookat[2] = 0.5
        self.viewer.cam.distance = 11.5
        self.viewer.cam.elevation = -65
        self.viewer.cam.azimuth = 270

    def get_states_sweep(self, nx, ny):
        x = np.linspace(-4, 4, nx)
        y = np.linspace(-4, 4, ny)
        xv, yv = np.meshgrid(x, y)
        states = np.stack((xv, yv), axis=2).reshape((-1, 2))
        return states

    def get_image_v(self, agent, qf, vf, obs, tau=None):
        nx, ny = (50, 50)
        x = np.linspace(-4, 4, nx)
        y = np.linspace(-4, 4, ny)
        xv, yv = np.meshgrid(x, y)

        sweep_obs = np.tile(obs.reshape((1, -1)), (nx * ny, 1))
        sweep_goal = np.stack((xv, yv), axis=2).reshape((-1, 2))
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
                sweep_actions = agent.eval_np(sweep_obs, sweep_goal, sweep_tau)
                v_vals = qf.eval_np(sweep_obs, sweep_actions, sweep_goal, sweep_tau)
            else:
                sweep_obs_goal = np.hstack((sweep_obs, sweep_goal))
                sweep_actions = agent.eval_np(sweep_obs_goal)
                v_vals = qf.eval_np(sweep_obs_goal, sweep_actions)
        if tau is not None:
            v_vals = -np.linalg.norm(v_vals, ord=qf.norm_order, axis=1)
        v_vals = v_vals.reshape((nx, ny))
        if self.v_func_heatmap_bounds is not None:
            vmin = self.v_func_heatmap_bounds[0]
            vmax = self.v_func_heatmap_bounds[1]
        else:
            vmin, vmax = None, None
        return self.get_image_plt(
            v_vals,
            vmin=vmin, vmax=vmax,
            draw_state=True, draw_goal=True,
        )

    def get_image_plt(
            self,
            vals,
            vmin=None, vmax=None,
            extent=[-4, 4, -4, 4],
            small_markers=False,
            draw_walls=True, draw_state=True, draw_goal=False, draw_subgoals=False,
            imsize=84
    ):
        fig, ax = plt.subplots()
        ax.set_ylim(extent[2:4])
        # ax.set_xlim(extent[0:2])
        ax.set_xlim([4, -4])
        ax.set_ylim(ax.get_ylim()[::-1])
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
            ball = plt.Circle(ob['state_observation'], self.ball_radius * marker_factor, color=color)
            ax.add_artist(ball)
        if draw_goal:
            goal = plt.Circle(ob['state_desired_goal'], self.ball_radius * marker_factor, color='green')
            ax.add_artist(goal)
        if draw_subgoals:
            if self.subgoals is not None:
                subgoal = plt.Circle(self.subgoals[0], (self.ball_radius + 0.1) * marker_factor, color='red')
            else:
                subgoal = None
            ax.add_artist(subgoal)
        if draw_walls:
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

class Wall(object, metaclass=abc.ABCMeta):
    def __init__(self, x_center, y_center, x_thickness, y_thickness, min_dist):
        self.min_x = x_center - x_thickness - min_dist
        self.max_x = x_center + x_thickness + min_dist
        self.min_y = y_center - y_thickness - min_dist
        self.max_y = y_center + y_thickness + min_dist

        self.endpoint1 = (x_center+x_thickness, y_center+y_thickness)
        self.endpoint2 = (x_center+x_thickness, y_center-y_thickness)
        self.endpoint3 = (x_center-x_thickness, y_center-y_thickness)
        self.endpoint4 = (x_center-x_thickness, y_center+y_thickness)

    def contains_point(self, point):
        return (self.min_x < point[0] < self.max_x) and (self.min_y < point[1] < self.max_y)

    def contains_points(self, points):
        return (self.min_x < points[:,0]) * (points[:,0] < self.max_x) \
               * (self.min_y < points[:,1]) * (points[:,1] < self.max_y)