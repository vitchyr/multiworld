from collections import OrderedDict
import logging

import numpy as np
from gym import spaces
from pygame import Color

from multiworld.core.multitask_env import MultitaskEnv
from multiworld.core.serializable import Serializable
from multiworld.envs.env_util import (
    get_stat_in_paths,
    create_stats_ordered_dict,
)
from multiworld.envs.pygame.pygame_viewer import PygameViewer


class Object(object):
    IDX_TO_COLOR = [
        Color('blue'),
        Color('green'),
        Color('red'),
        Color('purple'),
        Color('orange'),
        Color('yellow'),
    ]

    def __init__(self, position, radius, color, max_pos, min_pos):
        self.position = position
        self._color = color
        self._max_pos = max_pos
        self._min_pos = min_pos
        self._radius = radius
        self.target_position = None

    def distance_to_target(self):
        return np.linalg.norm(self.position - self.target_position)

    def draw(self, drawer, draw_target_position=True):
        drawer.draw_solid_circle(
            self.position,
            self._radius,
            self._color,
        )
        if draw_target_position:
            drawer.draw_circle(
                self.target_position,
                self._radius,
                self._color,
            )

    def distance(self, pos):
        return np.linalg.norm(self.position - pos)

    def move(self, velocity):
        self.position = np.clip(
            self.position + velocity,
            a_min=self._min_pos,
            a_max=self._max_pos,
        )


def draw_wall(drawer, wall):
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


class MultiObject2DPushEnv(MultitaskEnv, Serializable):
    """
    A simple env where a 'cursor' robot can push objects around.
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
            ball_radius=.75,
            object_radius=0.50,
            walls=None,
            fixed_goal=None,
            randomize_position_on_reset=True,
            fixed_reset=None,
            images_are_rgb=False,  # else black and white
            show_goal=True,
            expl_goal_sampler=None,
            eval_goal_sampler=None,
            use_fixed_reset_for_eval=False,
            num_objects=2,
            min_grab_distance=0.5,
            **kwargs
    ):
        walls = walls or []
        if fixed_goal is not None:
            fixed_goal = np.array(fixed_goal)
        if action_scale <= 0:
            raise ValueError("Invalid action scale: {}".format(
                action_scale
            ))
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
        self._show_goal = show_goal

        self._all_objects = [
            Object(
                position=np.zeros((2,)),
                color=Object.IDX_TO_COLOR[i],
                radius=ball_radius if i == 0 else object_radius,
                min_pos=-self.boundary_dist,
                max_pos=self.boundary_dist,
            )
            for i in range(num_objects + 1)
        ]
        self.min_grab_distance = min_grab_distance

        u = np.ones(3)
        self.action_space = spaces.Box(-u, u, dtype=np.float32)

        o = self.boundary_dist * np.ones(2 * (num_objects+1))
        self.obs_range = spaces.Box(-o, o, dtype='float32')
        self.observation_space = spaces.Dict([
            ('observation', self.obs_range),
            ('desired_goal', self.obs_range),
            ('achieved_goal', self.obs_range),
            ('state_observation', self.obs_range),
            ('state_desired_goal', self.obs_range),
            ('state_achieved_goal', self.obs_range),
        ])

        self._drawer = None
        self._render_drawer = None
        self.goal_sampling_mode = "test"
        self.fixed_reset = fixed_reset
        self.expl_goal_sampler = expl_goal_sampler
        self.eval_goal_sampler = eval_goal_sampler

        self.presampled_goals = None
        self.use_fixed_reset_for_eval = use_fixed_reset_for_eval

    @property
    def cursor(self):
        return self._all_objects[0]

    @property
    def objects(self):
        return self._all_objects[1:]

    def step(self, raw_action):
        velocities = raw_action[:2]
        grab = raw_action[2] > 0

        velocities = np.clip(velocities, a_min=-1, a_max=1) * self.action_scale
        old_position = self.cursor.position.copy()
        new_position = old_position + velocities
        orig_new_pos = new_position.copy()
        for wall in self.walls:
            new_position = wall.handle_collision(
                old_position, new_position
            )

        if sum(new_position != orig_new_pos) > 1:
            """
            Hack: sometimes you get caught on two walls at a time. If you
            process the input in the other direction, you might only get
            caught on one wall instead.
            """
            new_position = orig_new_pos.copy()
            for wall in self.walls[::-1]:
                new_position = wall.handle_collision(
                    old_position, new_position
                )

        if grab:
            grabbed_obj = self._grab_object()
            if grabbed_obj:
                grabbed_obj.move(velocities)

        self.cursor.move(new_position - old_position)

        ob = self._get_obs()
        reward = self.compute_reward(velocities, ob)
        info = self._get_info()
        done = False
        return ob, reward, done, info

    def _get_info(self):
        distance_to_target = self.cursor.distance_to_target()
        is_success = distance_to_target < self.target_radius
        info = {
            'distance_to_target': distance_to_target,
            'is_success': is_success,
        }
        for i, obj in enumerate(self.objects):
            obj_distance = obj.distance_to_target()
            info['distance_to_target_obj_{}'.format(i)] = (
                obj_distance
            )
        return info

    def reset(self):
        goal = self.sample_goal()['state_desired_goal']
        self._set_target_positions(goal)

        init_pos = np.zeros_like(self.obs_range.low)
        self._set_positions(init_pos)

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
        positions = self._get_positions()
        target_positions = self._get_target_positions()
        return dict(
            observation=positions.copy(),
            desired_goal=target_positions.copy(),
            achieved_goal=positions.copy(),
            state_observation=positions.copy(),
            state_desired_goal=target_positions.copy(),
            state_achieved_goal=positions.copy(),
        )

    def compute_rewards(self, actions, obs):
        achieved_goals = obs['state_achieved_goal']
        desired_goals = obs['state_desired_goal']
        d = np.linalg.norm(achieved_goals - desired_goals, axis=-1)
        if self.reward_type == "sparse":
            return -(d > self.target_radius).astype(np.float32)
        elif self.reward_type == "dense":
            return -d
        elif self.reward_type == "dense_l1":
            return -np.linalg.norm(achieved_goals - desired_goals, axis=-1,
                                   ord=1)
        elif self.reward_type == 'vectorized_dense':
            return -np.abs(achieved_goals - desired_goals)
        else:
            raise NotImplementedError()

    def get_diagnostics(self, paths, prefix=''):
        statistics = OrderedDict()
        for stat_name in [
            'distance_to_target',
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
        for i, obj in enumerate(self.objects):
            stat_name = 'distance_to_target_obj_{}'.format(i)
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
            'desired_goal': self._get_positions(),
            'state_desired_goal': self._get_target_positions(),
        }

    def set_goal(self, goal):
        self._set_target_positions(goal['state_desired_goal'])

    def sample_goals(self, batch_size):
        # if self.goal_sampling_mode == 'train' and self.expl_goal_sampler:
        #     return self.expl_goal_sampler(self, batch_size)
        # if self.goal_sampling_mode == 'test' and self.eval_goal_sampler:
        #     return self.eval_goal_sampler(self, batch_size)
        # assert self.goal_sampling_mode is None, "Invalid goal sampling mode: {}".format(self.goal_sampling_mode)

        if self.fixed_goal is not None:
            goals = state_goals = np.repeat(
                self.fixed_goal.copy()[None],
                batch_size,
                0
            )
        else:
            if self.presampled_goals is None:
                if len(self.walls) > 0:
                    presampled_goals = np.zeros((10000, self.obs_range.low.size))
                    for b in range(10000):
                        presampled_goals[b, :] = self._sample_position(
                            self.obs_range.low,
                            self.obs_range.high,
                        )
                else:
                    presampled_goals = np.random.uniform(
                        self.obs_range.low,
                        self.obs_range.high,
                        size=(10000, self.obs_range.low.size),
                    )
                self.presampled_goals = {
                    'desired_goal': presampled_goals,
                    'state_desired_goal': presampled_goals,
                }
            random_idxs = np.random.choice(len(list(self.presampled_goals.values())[0]), size=batch_size)
            goals = self.presampled_goals['desired_goal'][random_idxs]
            state_goals = self.presampled_goals['state_desired_goal'][random_idxs]
        return {
            'desired_goal': goals,
            'state_desired_goal': state_goals,
        }

    def _set_positions(self, positions):
        for i, obj in enumerate(self._all_objects):
            start_i = i*2
            end_i = i*2 + 2
            obj.position = positions[start_i:end_i]

    def _set_target_positions(self, target_positions):
        for i, obj in enumerate(self._all_objects):
            start_i = i*2
            end_i = i*2 + 2
            obj.target_position = target_positions[start_i:end_i]

    def _get_positions(self):
        return np.concatenate([
            obj.position for obj in self._all_objects
        ])

    def _get_target_positions(self):
        return np.concatenate([
            obj.target_position for obj in self._all_objects
        ])

    """Functions for ImageEnv wrapper"""

    def get_image(self, width=None, height=None):
        """Returns a black and white image"""
        if self._drawer is None:
            if width != height:
                raise NotImplementedError()
            self._drawer = PygameViewer(
                screen_width=width,
                screen_height=height,
                x_bounds=(-self.boundary_dist - self.ball_radius, self.boundary_dist + self.ball_radius),
                y_bounds=(-self.boundary_dist - self.ball_radius, self.boundary_dist + self.ball_radius),
                render_onscreen=self.render_onscreen,
            )
        self._draw(self._drawer)
        img = self._drawer.get_image()
        if self.images_are_rgb:
            return img.transpose((1, 0, 2))
        else:
            r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
            img = (-r + b).transpose().flatten()
            return img

    def set_to_goal(self, goal_dict):
        goal = goal_dict["state_desired_goal"]
        self._set_positions(goal)

    def get_env_state(self):
        return self._get_obs()

    def set_env_state(self, state):
        self._set_positions(state["state_observation"])
        self._set_target_positions(state["state_desired_goal"])

    def render(self, mode='human', close=False):
        if close:
            self._render_drawer = None
            return

        if self._render_drawer is None or self._render_drawer.terminated:
            self._render_drawer = PygameViewer(
                self.render_size,
                self.render_size,
                x_bounds=(-self.boundary_dist-self.ball_radius, self.boundary_dist+self.ball_radius),
                y_bounds=(-self.boundary_dist-self.ball_radius, self.boundary_dist+self.ball_radius),
                render_onscreen=True,
            )
        self._draw(self._render_drawer)
        self._render_drawer.tick(self.render_dt_msec)
        if mode != 'interactive':
            self._render_drawer.check_for_exit()

    def _draw(self, drawer):
        drawer.fill(Color('white'))
        for obj in self._all_objects:
            obj.draw(drawer, draw_target_position=self._show_goal)

        for wall in self.walls:
            draw_wall(drawer, wall)
        drawer.render()

    def get_diagnostics(self, paths, prefix=''):
        statistics = OrderedDict()
        for stat_name in [
            'distance_to_target_obj_{}'
            for i in range(len(self.objects))
        ] + ['distance_to_target']:
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

    def _grab_object(self):
        closest_object = None
        min_dis = None
        for obj in self.objects:
            distance = obj.distance(self.cursor.position)
            if (
                    (distance <= self.min_grab_distance)
                    and (closest_object is None or distance < min_dis)
            ):
                min_dis = distance
                closest_object = obj
        return closest_object
