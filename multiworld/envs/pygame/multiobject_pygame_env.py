from collections import OrderedDict
import logging

import numpy as np
from gym import spaces
from pygame import Color
import random
from multiworld.core.multitask_env import MultitaskEnv
from multiworld.core.serializable import Serializable
from multiworld.envs.env_util import (
    get_stat_in_paths,
    create_stats_ordered_dict,
)
from multiworld.envs.pygame.pygame_viewer import PygameViewer
from multiworld.envs.pygame.walls import VerticalWall, HorizontalWall


class Multiobj2DEnv(MultitaskEnv, Serializable):
    """
    A little 2D point whose life goal is to reach a target.
    """

    def __init__(
            self,
            render_dt_msec=0,
            action_l2norm_penalty=0,  # disabled for now
            render_onscreen=False,
            render_size=84,
            reward_type="dense",
            action_scale=1.0,
            target_radius=0.60,
            boundary_dist=4,
            ball_radius=0.50,
            include_colors_in_obs = False,
            walls=None,
            num_colors = 8,
            change_colors=True,
            fixed_colors = False,
            fixed_goal=None,
            randomize_position_on_reset=True,
            images_are_rgb=False,  # else black and white
            show_goal=True,
            use_env_labels = False,
            num_objects=1,
            include_white=False,
            **kwargs
    ):
        if walls is None:
            walls = []
        if walls is None:
            walls = []
        if fixed_goal is not None:
            fixed_goal = np.array(fixed_goal)
        if len(kwargs) > 0:
            LOGGER = logging.getLogger(__name__)
            LOGGER.log(logging.WARNING, "WARNING, ignoring kwargs:", kwargs)
        self.quick_init(locals())
        self.render_dt_msec = render_dt_msec
        self.action_l2norm_penalty = action_l2norm_penalty
        self.render_onscreen = render_onscreen
        self.render_size = render_size
        self.reward_type = reward_type
        self.action_scale = action_scale
        self.target_radius = target_radius
        self.boundary_dist = boundary_dist
        self.ball_radius = ball_radius
        self.walls = walls
        self.fixed_goal = fixed_goal
        self.randomize_position_on_reset = randomize_position_on_reset
        self.images_are_rgb = images_are_rgb
        self.show_goal = show_goal
        self.num_objects = num_objects
        self.max_target_distance = self.boundary_dist - self.target_radius
        self.change_colors = change_colors
        self.fixed_colors = fixed_colors
        self.num_colors = num_colors
        self._target_position = None
        self._position = np.zeros((2))
        self.include_white = include_white
        self.initial_pass = False
        self.white_passed = False

        if change_colors:
            self.randomize_colors()
        else:
            self.object_colors = [Color('blue')]

        u = np.ones(2)
        self.action_space = spaces.Box(-u, u, dtype=np.float32)
        self.include_colors_in_obs = include_colors_in_obs
        o = self.boundary_dist * np.ones(2)
        ohe_range = spaces.Box(np.zeros(self.num_colors), np.ones(self.num_colors), dtype='float32')
        if include_colors_in_obs and self.fixed_colors:
            high = np.concatenate((o, np.ones(self.num_colors)))
            low = np.concatenate((-o, np.zeros(self.num_colors)))
        else:
            high = o
            low = -o

        self.obs_range = spaces.Box(low, high, dtype='float32')
        self.use_env_labels = use_env_labels
        if self.use_env_labels:

            self.observation_space = spaces.Dict([
                ('label', ohe_range),
                ('observation', self.obs_range),
                ('desired_goal', self.obs_range),
                ('achieved_goal', self.obs_range),
                ('state_observation', self.obs_range),
                ('state_desired_goal', self.obs_range),
                ('state_achieved_goal', self.obs_range),
            ])

        else:
            self.observation_space = spaces.Dict([
                ('observation', self.obs_range),
                ('desired_goal', self.obs_range),
                ('achieved_goal', self.obs_range),
                ('state_observation', self.obs_range),
                ('state_desired_goal', self.obs_range),
                ('state_achieved_goal', self.obs_range),
            ])

        self.drawer = None
        self.render_drawer = None

        self.color_index = 0
        self.colors = [Color('green'), Color('red'), Color('blue'), Color('black'), Color('purple'), Color('brown'), Color('pink'), Color('orange'), Color('grey'), Color('yellow')]

    def randomize_colors(self):
        self.object_colors = []
        rgbs = np.random.randint(0, 256, (self.num_objects, 3))

        for i in range(self.num_objects):
            if self.fixed_colors:
                self.object_colors.append(self.colors[self.color_index])
                self.color_index = (self.color_index + 1) % 10
            else:
                rgb = map(int, rgbs[i, :])
                self.object_colors.append(Color(*rgb, 255))

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
        is_success = distance_to_target < self.target_radius

        ob = self._get_obs()
        reward = self.compute_reward(velocities, ob)
        info = {
            'radius': self.target_radius,
            'target_position': self._target_position,
            'distance_to_target': distance_to_target,
            'velocity': velocities,
            'speed': np.linalg.norm(velocities),
            'is_success': is_success,
        }
        done = False
        return ob, reward, done, info

    def reset(self):
        if self.white_passed:
            self.include_white = False
        if self.change_colors:
            self.randomize_colors()
        self._target_position = self.sample_goal()['state_desired_goal']
        if self.randomize_position_on_reset:
            self._position = self._sample_position(
                self.obs_range.low,
                self.obs_range.high,
            )
        if self.initial_pass:
            self.white_passed = True
        self.initial_pass = True
        return self._get_obs()

    def _position_inside_wall(self, pos):
        for wall in self.walls:
            if wall.contains_point(pos):
                return True
        return False

    def _sample_position(self, low, high):
        pos = np.random.uniform(low, high)
        while self._position_inside_wall(pos) is True:
            pos = np.random.uniform(low, high)
        return pos

    def _get_obs(self):
        obs = self._position.copy()
        if self.use_env_labels:
            return dict(
                observation=obs,
                label = ohe,
                desired_goal=self._target_position.copy(),
                achieved_goal=self._position.copy(),
                state_observation=self._position.copy(),
                state_desired_goal=self._target_position.copy(),
                state_achieved_goal=self._position.copy(),
            )
        else:
            return dict(
                observation=obs,
                desired_goal=self._target_position.copy(),
                achieved_goal=self._position.copy(),
                state_observation=self._position.copy(),
                state_desired_goal=self._target_position.copy(),
                state_achieved_goal=self._position.copy(),
            )

    def compute_rewards(self, actions, obs):
        achieved_goals = obs['state_achieved_goal']
        desired_goals = obs['state_desired_goal']
        d = np.linalg.norm(achieved_goals - desired_goals, axis=-1)
        if self.reward_type == "sparse":
            return -(d > self.target_radius).astype(np.float32)
        elif self.reward_type == "dense":
            return -d
        elif self.reward_type == 'vectorized_dense':
            return -np.abs(achieved_goals - desired_goals)
        else:
            raise NotImplementedError()

    def get_diagnostics(self, paths, prefix=''):
        statistics = OrderedDict()
        for stat_name in [
            'radius',
            'target_position',
            'distance_to_target',
            'velocity',
            'speed',
            'is_success',
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
        self._position[0] = pos[0]
        self._position[1] = pos[1]

    """Functions for ImageEnv wrapper"""

    def get_image(self, width=None, height=None):
        """Returns a black and white image"""
        if self.drawer is None:
            if width != height:
                raise NotImplementedError()
            self.drawer = PygameViewer(
                screen_width=width,
                screen_height=height,
                x_bounds=(-self.boundary_dist - self.ball_radius, self.boundary_dist + self.ball_radius),
                y_bounds=(-self.boundary_dist - self.ball_radius, self.boundary_dist + self.ball_radius),
                render_onscreen=self.render_onscreen,
            )
        self.draw(self.drawer, False)
        img = self.drawer.get_image()
        if self.images_are_rgb:
            return img.transpose((1, 0, 2))
        else:
            r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
            img = (-r + b).transpose().flatten()
            return img

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

    def draw(self, drawer, tick):
        # if self.drawer is not None:
        #     self.drawer.fill(Color('white'))
        # if self.render_drawer is not None:
        #     self.render_drawer.fill(Color('white'))
        drawer.fill(Color('white'))
        if self.show_goal:
            drawer.draw_solid_circle(
                self._target_position,
                self.target_radius,
                Color('green'),
            )
        if not self.include_white:
            drawer.draw_solid_circle(
                self._position,
                self.ball_radius,
                self.object_colors[0],
            )

        for wall in self.walls:
            drawer.draw_segment(
                wall.endpoint1,
                wall.endpoint2,
                Color('black'),
            )
            drawer.draw_segment(
                wall.endpoint2,
                wall.endpoint3,
                Color('black'),
            )
            drawer.draw_segment(
                wall.endpoint3,
                wall.endpoint4,
                Color('black'),
            )
            drawer.draw_segment(
                wall.endpoint4,
                wall.endpoint1,
                Color('black'),
            )

        drawer.render()
        if tick:
            drawer.tick(self.render_dt_msec)

    def render(self, close=False):
        if close:
            self.render_drawer = None
            return

        if self.render_drawer is None or self.render_drawer.terminated:
            self.render_drawer = PygameViewer(
                self.render_size,
                self.render_size,
                x_bounds=(-self.boundary_dist-self.ball_radius, self.boundary_dist+self.ball_radius),
                y_bounds=(-self.boundary_dist-self.ball_radius, self.boundary_dist+self.ball_radius),
                render_onscreen=True,
            )
        self.draw(self.render_drawer, True)

    def get_diagnostics(self, paths, prefix=''):
        statistics = OrderedDict()
        for stat_name in [
            'distance_to_target',
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


class Multiobj2DWallEnv(Multiobj2DEnv):
    """Point2D with walls"""

    def __init__(
            self,
            wall_shape="",
            wall_thickness=1.0,
            inner_wall_max_dist=1,
            change_walls=False,
            **kwargs
    ):
        self.quick_init(locals())
        super().__init__(**kwargs)
        self.inner_wall_max_dist = inner_wall_max_dist
        self.wall_shape = wall_shape
        self.wall_shapes = ["right", "left", "bottom", "top"]
        self.wall_thickness = wall_thickness
        self.change_walls = change_walls

        if self.change_walls:
            self.randomize_walls()
        else:
            self.fixed_wall(wall_shape)

    def randomize_walls(self):
        self.walls = []
        random.shuffle(self.wall_shapes)
        for w in self.wall_shapes[:3]:
            if np.random.uniform() < 0.333:
                self.add_wall(w)

    def add_wall(self, wall):
        if wall == "right":
            # Right wall
            self.walls.append(VerticalWall(
                self.ball_radius,
                self.inner_wall_max_dist,
                -self.inner_wall_max_dist,
                self.inner_wall_max_dist,
            ))
        if wall == "left":# Left wall
            self.walls.append(VerticalWall(
                self.ball_radius,
                -self.inner_wall_max_dist,
                -self.inner_wall_max_dist,
                self.inner_wall_max_dist,
            ))
        if wall == "bottom":# Bottom wall
            self.walls.append(HorizontalWall(
                self.ball_radius,
                self.inner_wall_max_dist,
                -self.inner_wall_max_dist,
                self.inner_wall_max_dist,
            ))
        if wall == "top":
            self.walls.append(HorizontalWall(
                self.ball_radius,
                -self.inner_wall_max_dist,
                -self.inner_wall_max_dist,
                self.inner_wall_max_dist,
            ))

    def fixed_wall(self, wall_shape):
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

    # def reset(self):
    #     if self.change_colors:
    #         if self.fixed_colors:
    #             self.color_index = (self.color_index + 1) % 10
    #             self.pm_color = self.colors[self.color_index]
    #         else:
    #             rgbs = np.random.randint(0, 256, (1, 3))
    #             rgb = map(int, rgbs[0, :])
    #             self.pm_color = Color(*rgb, 255)
    #     else:
    #         pm_color = Color('blue')
    #     if self.change_walls:
    #         self.randomize_walls()
    #     if self.randomize_position_on_reset:
    #         self._position = self._sample_position(
    #             self.obs_range.low,
    #             self.obs_range.high)
    #     self._target_position = self.sample_goal()['state_desired_goal']
    #     return self._get_obs()

    def reset(self):
        if self.white_passed:
            self.include_white = False
        if self.change_colors:
            self.randomize_colors()
        if self.change_walls:
            self.randomize_walls()
        self._target_position = self.sample_goal()['state_desired_goal']
        if self.randomize_position_on_reset:
            self._position = self._sample_position(
                self.obs_range.low,
                self.obs_range.high,
            )
        if self.initial_pass:
            self.white_passed = True
        self.initial_pass = True
        return self._get_obs()


if __name__ == "__main__":
    import gym
    import cv2
    e = Multiobj2DEnv(
        render_onscreen=True,
        show_goal=False,
        fixed_colors=True,
        num_colors=7
    )
    from multiworld.core.image_env import ImageEnv
    from multiworld.envs.mujoco.cameras import sawyer_init_camera_zoomed_in

    e.reset()
    for i in range(1000):
        img = e.get_image(100, 100)
        if i % 20 == 0:
            print('reset')
            e.reset()
        cv2.imshow('im', img)
        cv2.waitKey(10)

