from collections import OrderedDict

import numpy as np
from collections import OrderedDict
from gym import spaces
from pygame import Color

from multiworld.envs.env_util import get_stat_in_paths, \
    create_stats_ordered_dict
from multiworld.core.image_env import ImageEnv
from multiworld.core.multitask_env import MultitaskEnv
from multiworld.core.serializable import Serializable
from multiworld.envs.env_util import (
    get_stat_in_paths,
    create_stats_ordered_dict,
)
from multiworld.envs.pygame.pygame_viewer import PygameViewer
from multiworld.envs.pygame.walls import VerticalWall, HorizontalWall

from railrl.pythonplusplus import identity

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from railrl.torch import pytorch_util as ptu


class Point2DEnv(MultitaskEnv, Serializable):
    """
    A little 2D point whose life goal is to reach a target.
    """

    def __init__(
            self,
            render_dt_msec=0,
            action_l2norm_penalty=0,  # disabled for now
            render_onscreen=False,
            render_size=200,
            render_target=True,
            reward_type="dense",
            norm_order=2,
            action_scale=1.0,
            target_radius=0.50,
            boundary_dist=4,
            ball_radius=0.50,
            walls=[],
            fixed_goal=None,
            goal_low=None,
            goal_high=None,
            ball_low=None,
            ball_high=None,
            randomize_position_on_reset=True,
            sample_realistic_goals=False,
            images_are_rgb=False,  # else black and white
            show_goal=True,
            **kwargs
    ):
        if walls is None:
            walls = []
        if len(kwargs) > 0:
            import logging
            LOGGER = logging.getLogger(__name__)
            LOGGER.log(logging.WARNING, "WARNING, ignoring kwargs:", kwargs)
        self.quick_init(locals())
        self.render_dt_msec = render_dt_msec
        self.action_l2norm_penalty = action_l2norm_penalty
        self.render_onscreen = render_onscreen
        self.render_size = render_size
        self.render_target = render_target
        self.reward_type = reward_type
        self.norm_order = norm_order
        self.action_scale = action_scale
        self.target_radius = target_radius
        self.boundary_dist = boundary_dist
        self.ball_radius = ball_radius
        self.walls = walls
        self.fixed_goal = fixed_goal
        self.randomize_position_on_reset = randomize_position_on_reset
        self.sample_realistic_goals = sample_realistic_goals
        if self.fixed_goal is not None:
            self.fixed_goal = np.array(self.fixed_goal)
        self.images_are_rgb = images_are_rgb
        self.show_goal = show_goal

        self.max_target_distance = self.boundary_dist - self.target_radius

        self._target_position = None
        self._position = np.zeros((2))

        u = np.ones(2)
        self.action_space = spaces.Box(-u, u, dtype=np.float32)

        o = self.boundary_dist * np.ones(2)
        self.obs_range = spaces.Box(-o, o, dtype='float32')

        if goal_low is None:
            goal_low = -o
        if goal_high is None:
            goal_high = o
        self.goal_range = spaces.Box(np.array(goal_low), np.array(goal_high), dtype='float32')

        if ball_low is None:
            ball_low = -o
        if ball_high is None:
            ball_high = o
        self.ball_range = spaces.Box(np.array(ball_low), np.array(ball_high), dtype='float32')

        self.observation_space = spaces.Dict([
            ('observation', self.obs_range),
            ('desired_goal', self.goal_range),
            ('achieved_goal', self.obs_range),
            ('state_observation', self.obs_range),
            ('state_desired_goal', self.goal_range),
            ('state_achieved_goal', self.obs_range),
        ])

        self.drawer = None
        self.subgoals = None

    def step(self, velocities):
        assert self.action_scale <= 1.0
        velocities = np.clip(velocities, a_min=-1, a_max=1) * self.action_scale
        new_position = self._position + velocities
        for wall in self.walls:
            new_position = wall.handle_collision(
                self._position, new_position
            )
        self._position = new_position
        self._position = np.clip(
            self._position,
            a_min=-self.boundary_dist,
            a_max=self.boundary_dist,
        )
        distance_to_target = np.linalg.norm(
            self._position - self._target_position
        )
        below_wall = self._position[1] >= 2.0
        is_success = distance_to_target < 1.0
        is_success_2 = distance_to_target < 1.5
        is_success_3 = distance_to_target < 1.75

        ob = self._get_obs()
        reward = self.compute_reward(velocities, ob)
        info = {
            'radius': self.target_radius,
            'target_position': self._target_position,
            'distance_to_target': distance_to_target,
            'velocity': velocities,
            'speed': np.linalg.norm(velocities),
            'is_success': is_success,
            'is_success_2': is_success_2,
            'is_success_3': is_success_3,
            'below_wall': below_wall,
        }
        done = False
        return ob, reward, done, info

    def initialize_camera(self, camera):
        pass

    def reset(self):
        self._target_position = self.sample_goal_for_rollout()['state_desired_goal']
        if self.randomize_position_on_reset:
            self._position = self._sample_position(self.ball_range.low, self.ball_range.high, realistic=True)

        self.subgoals = None
        return self._get_obs()

    def _position_outside_arena(self, pos):
        return not self.obs_range.contains(pos)

    def _position_inside_wall(self, pos):
        for wall in self.walls:
            if wall.contains_point(pos):
                return True
        return False

    def realistic_state_np(self, state):
        return not self._position_inside_wall(state)

    def realistic_goals(self, g, use_double=False):
        collision = None
        for wall in self.walls:
            wall_collision = wall.contains_points_pytorch(g)
            wall_collision_result = wall_collision[:, 0]
            for i in range(wall_collision.shape[1]):
                wall_collision_result = wall_collision_result * wall_collision[:, i]

            if collision is None:
                collision = wall_collision_result
            else:
                collision = collision | wall_collision_result

        result = collision ^ 1

        # g_np = ptu.get_numpy(g)
        # for i in range(g_np.shape[0]):
        #     print(not self._position_inside_wall(g_np[i]), ptu.get_numpy(result[i]))

        return result.float()

    def _sample_position(self, low, high, realistic=True):
        pos = np.random.uniform(low, high)
        if realistic:
            while self._position_inside_wall(pos) is True:
                pos = np.random.uniform(low, high)
        return pos

    def _get_obs(self):
        return dict(
            observation=self._position.copy(),
            desired_goal=self._target_position.copy(),
            achieved_goal=self._position.copy(),
            state_observation=self._position.copy(),
            state_desired_goal=self._target_position.copy(),
            state_achieved_goal=self._position.copy(),
        )

    def compute_rewards(self, actions, obs, prev_obs=None):
        achieved_goals = obs['state_achieved_goal']
        desired_goals = obs['state_desired_goal']
        d = np.linalg.norm(achieved_goals - desired_goals, ord=self.norm_order, axis=-1)
        if self.reward_type == "sparse":
            return -(d > self.target_radius).astype(np.float32)
        elif self.reward_type == "dense":
            return -d
        elif self.reward_type == 'vectorized_dense':
            return -np.abs(achieved_goals - desired_goals)

    def get_diagnostics(self, paths, prefix=''):
        statistics = OrderedDict()
        for stat_name in [
            'distance_to_target',
            'below_wall',
            'is_success',
            'is_success_2',
            'is_success_3',
            'velocity',
            'speed',
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

    def get_goal(self):
        return {
            'desired_goal': self._target_position.copy(),
            'state_desired_goal': self._target_position.copy(),
        }

    def set_goal(self, goal):
        self._target_position = goal['state_desired_goal']

    def sample_goal_for_rollout(self):
        if not self.fixed_goal is None:
            goal = np.copy(self.fixed_goal)
        else:
            goal = self._sample_position(self.goal_range.low,
                                         self.goal_range.high,
                                         realistic=self.sample_realistic_goals)
        return {
            'desired_goal': goal,
            'state_desired_goal': goal,
        }

    def sample_goals(self, batch_size):
        # goals = np.random.uniform(
        #     self.obs_range.low,
        #     self.obs_range.high,
        #     size=(batch_size, self.obs_range.low.size),
        # )
        goals = np.array(
            [
                self._sample_position(self.obs_range.low, self.obs_range.high, realistic=self.sample_realistic_goals)
                for _ in range(batch_size)
            ]
        )
        return {
            'desired_goal': goals,
            'state_desired_goal': goals,
        }

    def set_position(self, pos):
        self._position[0] = pos[0]
        self._position[1] = pos[1]

    """Functions for ImageEnv wrapper"""

    def get_image(self, width=None, height=None):
        """Returns a black and white image"""
        if width is not None:
            if width != height:
                raise NotImplementedError()
            if width != self.render_size:
                self.drawer = PygameViewer(
                    screen_width=width,
                    screen_height=height,
                    x_bounds=(-self.boundary_dist - self.ball_radius, self.boundary_dist + self.ball_radius),
                    y_bounds=(-self.boundary_dist - self.ball_radius, self.boundary_dist + self.ball_radius),
                    render_onscreen=self.render_onscreen,
                )
                self.render_size = width
        self.render()
        img = self.drawer.get_image()
        if self.images_are_rgb:
            return img.transpose((1, 0, 2))
        else:
            r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
            img = (-r + b)
            return img

    def update_subgoals(self, subgoals):
        self.subgoals = subgoals

    def set_to_goal(self, goal_dict):
        goal = goal_dict["state_desired_goal"]
        self._position = goal
        self._target_position = goal

    def get_env_state(self):
        return self._get_obs()

    def set_env_state(self, state):
        position = state["state_observation"]
        goal = state["state_desired_goal"]
        self._position = position
        self._target_position = goal

    def get_states_sweep(self, nx, ny):
        x = np.linspace(-4, 4, nx)
        y = np.linspace(-4, 4, ny)
        xv, yv = np.meshgrid(x, y)
        states = np.stack((xv, yv), axis=2).reshape((-1, 2))
        return states

    def render(self, close=False):
        if close:
            self.drawer = None
            return

        if self.drawer is None or self.drawer.terminated:
            self.drawer = PygameViewer(
                self.render_size,
                self.render_size,
                x_bounds=(-self.boundary_dist-self.ball_radius, self.boundary_dist+self.ball_radius),
                y_bounds=(-self.boundary_dist-self.ball_radius, self.boundary_dist+self.ball_radius),
                render_onscreen=self.render_onscreen,
            )
        self.drawer.fill(Color('white'))
        if self.render_target:
            self.drawer.draw_solid_circle(
                self._target_position,
                self.target_radius,
                Color('green'),
            )
        self.drawer.draw_solid_circle(
            self._position,
            self.ball_radius,
            Color('blue'),
        )

        if self.subgoals is not None:
            for goal in self.subgoals:
                self.drawer.draw_solid_circle(
                    goal,
                    self.ball_radius + 0.1,
                    Color('red'),
                )
            for p1, p2 in zip(np.concatenate(([self._position], self.subgoals[:-1]), axis=0), self.subgoals):
                self.drawer.draw_segment(p1, p2, Color(100, 0, 0, 10))

        for wall in self.walls:
            self.drawer.draw_segment(
                wall.endpoint1,
                wall.endpoint2,
                Color('black'),
            )
            self.drawer.draw_segment(
                wall.endpoint2,
                wall.endpoint3,
                Color('black'),
            )
            self.drawer.draw_segment(
                wall.endpoint3,
                wall.endpoint4,
                Color('black'),
            )
            self.drawer.draw_segment(
                wall.endpoint4,
                wall.endpoint1,
                Color('black'),
            )

        self.drawer.render()
        self.drawer.tick(self.render_dt_msec)

    """Static visualization/utility methods"""

    @staticmethod
    def true_model(state, action):
        velocities = np.clip(action, a_min=-1, a_max=1)
        position = state
        new_position = position + velocities
        return np.clip(
            new_position,
            a_min=-Point2DEnv.boundary_dist,
            a_max=Point2DEnv.boundary_dist,
        )

    @staticmethod
    def true_states(state, actions):
        real_states = [state]
        for action in actions:
            next_state = Point2DEnv.true_model(state, action)
            real_states.append(next_state)
            state = next_state
        return real_states

    @staticmethod
    def plot_trajectory(ax, states, actions, goal=None):
        assert len(states) == len(actions) + 1
        x = states[:, 0]
        y = -states[:, 1]
        num_states = len(states)
        plasma_cm = plt.get_cmap('plasma')
        for i, state in enumerate(states):
            color = plasma_cm(float(i) / num_states)
            ax.plot(state[0], -state[1],
                    marker='o', color=color, markersize=10,
                    )

        actions_x = actions[:, 0]
        actions_y = -actions[:, 1]

        ax.quiver(x[:-1], y[:-1], x[1:] - x[:-1], y[1:] - y[:-1],
                  scale_units='xy', angles='xy', scale=1, width=0.005)
        ax.quiver(x[:-1], y[:-1], actions_x, actions_y, scale_units='xy',
                  angles='xy', scale=1, color='r',
                  width=0.0035, )
        ax.plot(
            [
                -Point2DEnv.boundary_dist,
                -Point2DEnv.boundary_dist,
            ],
            [
                Point2DEnv.boundary_dist,
                -Point2DEnv.boundary_dist,
            ],
            color='k', linestyle='-',
        )
        ax.plot(
            [
                Point2DEnv.boundary_dist,
                -Point2DEnv.boundary_dist,
            ],
            [
                Point2DEnv.boundary_dist,
                Point2DEnv.boundary_dist,
            ],
            color='k', linestyle='-',
        )
        ax.plot(
            [
                Point2DEnv.boundary_dist,
                Point2DEnv.boundary_dist,
            ],
            [
                Point2DEnv.boundary_dist,
                -Point2DEnv.boundary_dist,
            ],
            color='k', linestyle='-',
        )
        ax.plot(
            [
                Point2DEnv.boundary_dist,
                -Point2DEnv.boundary_dist,
            ],
            [
                -Point2DEnv.boundary_dist,
                -Point2DEnv.boundary_dist,
            ],
            color='k', linestyle='-',
        )

        if goal is not None:
            ax.plot(goal[0], -goal[1], marker='*', color='g', markersize=15)
        ax.set_ylim(
            -Point2DEnv.boundary_dist - 1,
            Point2DEnv.boundary_dist + 1,
        )
        ax.set_xlim(
            -Point2DEnv.boundary_dist - 1,
            Point2DEnv.boundary_dist + 1,
        )

    def initialize_camera(self, init_fctn):
        pass


class Point2DWallEnv(Point2DEnv):
    """Point2D with walls"""

    def __init__(
            self,
            wall_shape="",
            wall_thickness=0.0,
            inner_wall_max_dist=1,
            **kwargs
    ):
        self.quick_init(locals())
        super().__init__(**kwargs)
        self.inner_wall_max_dist = inner_wall_max_dist
        self.wall_shape = wall_shape
        self.wall_thickness = wall_thickness

        if wall_shape == "u":
            self.walls = [
                # Right wall
                VerticalWall(
                    self.ball_radius,
                    self.inner_wall_max_dist,
                    -self.inner_wall_max_dist,
                    self.inner_wall_max_dist,
                ),
                # Left wall
                VerticalWall(
                    self.ball_radius,
                    -self.inner_wall_max_dist,
                    -self.inner_wall_max_dist,
                    self.inner_wall_max_dist,
                ),
                # Bottom wall
                HorizontalWall(
                    self.ball_radius,
                    self.inner_wall_max_dist,
                    -self.inner_wall_max_dist,
                    self.inner_wall_max_dist,
                )
            ]
        if wall_shape == "-" or wall_shape == "h":
            self.walls = [
                HorizontalWall(
                    self.ball_radius,
                    self.inner_wall_max_dist,
                    -self.inner_wall_max_dist,
                    self.inner_wall_max_dist,
                )
            ]
        if wall_shape == "--":
            self.walls = [
                HorizontalWall(
                    self.ball_radius,
                    0,
                    -self.inner_wall_max_dist,
                    self.inner_wall_max_dist,
                )
            ]
        if wall_shape == "big-u":
            self.walls = [
                VerticalWall(
                    self.ball_radius,
                    self.inner_wall_max_dist*2,
                    -self.inner_wall_max_dist*2,
                    self.inner_wall_max_dist,
                    self.wall_thickness
                ),
                # Left wall
                VerticalWall(
                    self.ball_radius,
                    -self.inner_wall_max_dist*2,
                    -self.inner_wall_max_dist*2,
                    self.inner_wall_max_dist,
                    self.wall_thickness
                ),
                # Bottom wall
                HorizontalWall(
                    self.ball_radius,
                    self.inner_wall_max_dist,
                    -self.inner_wall_max_dist*2,
                    self.inner_wall_max_dist*2,
                    self.wall_thickness
                ),
            ]
        if wall_shape == "easy-u":
            self.walls = [
                VerticalWall(
                    self.ball_radius,
                    self.inner_wall_max_dist*2,
                    -self.inner_wall_max_dist*0.5,
                    self.inner_wall_max_dist,
                    self.wall_thickness
                ),
                # Left wall
                VerticalWall(
                    self.ball_radius,
                    -self.inner_wall_max_dist*2,
                    -self.inner_wall_max_dist*0.5,
                    self.inner_wall_max_dist,
                    self.wall_thickness
                ),
                # Bottom wall
                HorizontalWall(
                    self.ball_radius,
                    self.inner_wall_max_dist,
                    -self.inner_wall_max_dist*2,
                    self.inner_wall_max_dist*2,
                    self.wall_thickness
                ),
            ]
        if wall_shape == "flappy-bird":
            self.walls = [
                VerticalWall(
                    self.ball_radius,
                    self.inner_wall_max_dist*1.667,
                    -self.inner_wall_max_dist*0.5,
                    self.inner_wall_max_dist*4,
                    self.wall_thickness
                ),
                VerticalWall(
                    self.ball_radius,
                    -self.inner_wall_max_dist*1.667,
                    -self.inner_wall_max_dist*4,
                    self.inner_wall_max_dist*0.5,
                    self.wall_thickness
                ),
            ]
        if wall_shape == "big-h":
            self.walls = [
                # Bottom wall
                HorizontalWall(
                    self.ball_radius,
                    self.inner_wall_max_dist,
                    -self.inner_wall_max_dist*2,
                    self.inner_wall_max_dist*2,
                ),
            ]
        if wall_shape == "box":
            self.walls = [
                # Bottom wall
                VerticalWall(
                    self.ball_radius,
                    0,
                    0,
                    0,
                    self.wall_thickness
                ),
            ]
        if wall_shape == "none":
            self.walls = []

    def generate_expert_subgoals(self, num_subprobs):
        def avg(p1, p2):
            return [(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2]
        ob_and_goal = self._get_obs()
        ob = ob_and_goal['state_observation']
        goal = ob_and_goal['state_desired_goal']

        subgoals = []
        if self.wall_shape == "big-u" or self.wall_shape == "easy-u":
            if num_subprobs == 2:
                if goal[0] <= 0:
                    subgoals += [(-3, -3)]
                else:
                    subgoals += [(3, -3)]
            elif num_subprobs in [4, 8]:
                subgoals += [
                    [1.0, -3.25],
                    [3.25, -2.0],
                    [3.25, 1.5]
                ]

                if goal[0] <= 0:
                    for subgoal in subgoals:
                        subgoal[0] *= -1

                if num_subprobs == 8:
                    subgoals_1 = subgoals
                    subgoals_2 = []

                    for first, second in zip([ob] + subgoals_1 + [goal], subgoals_1 + [goal]):
                        subgoals_2.append(avg(first, second))

                    if goal[0] <= 0:
                        subgoals_2[1] = [-2.5, -3.25]
                    else:
                        subgoals_2[1] = [2.5, -3.25]
                    subgoals_2[3][1] = max(subgoals_2[3][1], 2.25)

                    subgoals = [subgoals_2[0]] + [item for t in zip(subgoals_1, subgoals_2[1:]) for item in t]
            elif num_subprobs == 6:
                subgoals += [
                    [0.5, -2.0],
                    [2.5, -3.25],
                    [3.25, -1.25],
                    [3.25, 0.75],
                ]

                if goal[0] <= 0:
                    for subgoal in subgoals:
                        subgoal[0] *= -1

                subgoals.append(avg(subgoals[-1], goal))
                subgoals[-1][1] = max(subgoals[-1][1], 2.25)

        if len(subgoals) == 0:
            subgoals = np.tile(goal, num_subprobs-1).reshape(num_subprobs-1, -1)

        return np.array(subgoals)

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
        return self.get_image_plt(v_vals, vmin=-2.0, vmax=0.0)

    def get_image_rew(self, obs):
        nx, ny = (50, 50)
        x = np.linspace(-4, 4, nx)
        y = np.linspace(-4, 4, ny)
        xv, yv = np.meshgrid(x, y)

        sweep_obs = np.tile(obs.reshape((1, -1)), (nx * ny, 1))
        sweep_goal = np.stack((xv, yv), axis=2).reshape((-1, 2))
        rew_vals = -np.linalg.norm(sweep_obs - sweep_goal, ord=self.norm_order, axis=-1).reshape((nx, ny))
        return self.get_image_plt(rew_vals)

    def get_image_realistic_rew(self):
        nx, ny = (50, 50)
        x = np.linspace(-4, 4, nx)
        y = np.linspace(-4, 4, ny)
        xv, yv = np.meshgrid(x, y)

        sweep_goal = np.stack((xv, yv), axis=2).reshape((-1, 2))
        rew_vals = ptu.get_numpy(self.realistic_goals(sweep_goal)).reshape((nx, ny))
        return self.get_image_plt(rew_vals)

    def get_image_plt(self,
                      vals,
                      vmin=None, vmax=None,
                      extent=[-4, 4, -4, 4],
                      small_markers=False,
                      draw_walls=True, draw_state=True, draw_goal=False, draw_subgoals=False,
                      imsize=None):
        fig, ax = plt.subplots()
        ax.set_ylim(extent[2:4])
        ax.set_xlim(extent[0:2])
        ax.set_ylim(ax.get_ylim()[::-1])
        DPI = fig.get_dpi()
        if imsize is None:
            fig.set_size_inches(self.render_size / float(DPI), self.render_size / float(DPI))
        else:
            fig.set_size_inches(imsize / float(DPI), imsize / float(DPI))

        marker_factor = 0.60
        if small_markers:
            marker_factor = 0.10

        if draw_state:
            if small_markers:
                color = 'cyan'
            else:
                color = 'blue'
            ball = plt.Circle(self._position, self.ball_radius * marker_factor, color=color)
            ax.add_artist(ball)
        if draw_goal:
            goal = plt.Circle(self._target_position, self.target_radius * marker_factor, color='green')
            ax.add_artist(goal)
        if draw_subgoals:
            if self.subgoals is not None:
                subgoal = plt.Circle(self.subgoals[0], (self.ball_radius + 0.1) * marker_factor, color='red')
            else:
                subgoal = None
            ax.add_artist(subgoal)
        if draw_walls:
            for wall in self.walls:
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

class PointmassExpertPolicy:
    """
    """
    def __init__(
            self,
            env,
            base_policy,
            prob_expert=0.5,
    ):
        self.env = env
        self.base_policy = base_policy
        self.prob_expert = prob_expert

    def reset(self):
        pass

    @staticmethod
    def point_in_box(point, low, high):
        return spaces.Box(np.array(low), np.array(high), dtype='float32').contains(point)

    def get_action(self, obs, goal_high_level, tau_high_level):
        action = [0, 0]
        if np.random.uniform(0, 1) <= self.prob_expert:
            br = self.env.ball_radius + self.env.wall_thickness
            if self.env.wall_shape == "big-u":
                if self.point_in_box(obs, low=[-2+br, -2-br], high=[2-br, 1+br]):
                    action = [0, -1]
                elif self.point_in_box(obs, low=[0, -4], high=[2+br, -2-br]):
                    action = [1, 0]
                elif self.point_in_box(obs, low=[-2-br, -4], high=[0, -2-br]):
                    action = [-1, 0]
                elif self.point_in_box(obs, low=[2+br, -4], high=[4, 1+br]):
                    action = [0, 1]
                elif self.point_in_box(obs, low=[-4, -4], high=[-2-br, 1+br]):
                    action = [0, 1]
                elif self.point_in_box(obs, low=[-4, 1+br], high=[4, 4]):
                    action = goal_high_level - obs
                    if np.linalg.norm(action) > 1:
                        action = action / np.linalg.norm(action)
            elif self.env.wall_shape == "big-h":
                if self.point_in_box(obs, low=[-2-br, -4], high=[0, 1+br]):
                    action = [-1, 0]
                elif self.point_in_box(obs, low=[0, -4], high=[2+br, 1+br]):
                    action = [1, 0]
                elif self.point_in_box(obs, low=[-4, -4], high=[-2-br, 1+br]):
                    action = [0, 1]
                elif self.point_in_box(obs, low=[2+br, -4], high=[4, 1+br]):
                    action = [0, 1]
                elif self.point_in_box(obs, low=[-4, 1+br], high=[4, 4]):
                    action = goal_high_level - obs
                    if np.linalg.norm(action) > 1:
                        action = action / np.linalg.norm(action)
            elif self.env.wall_shape == "none":
                action = goal_high_level - obs
                if np.linalg.norm(action) > 1:
                    action = action / np.linalg.norm(action)
        else:
            action = self.base_policy.eval_np(obs[None], goal_high_level[None], tau_high_level[None])[0]

        return np.array(action), {}


if __name__ == "__main__":
    # e = Point2DEnv()
    import matplotlib.pyplot as plt

    # e = Point2DWallEnv("-", render_size=84)
    e = ImageEnv(Point2DWallEnv(wall_shape="u", render_size=84))
    for i in range(10):
        e.reset()
        for j in range(50):
            e.step(np.random.rand(2))
            e.render()
            im = e.get_image()
