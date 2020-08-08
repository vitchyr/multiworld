from collections import OrderedDict, defaultdict
import random

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

import matplotlib.pyplot as plt


class Object(object):
    IDX_TO_COLOR = [
        Color('blue'),
        Color('green'),
        Color('red'),
        Color('purple'),
        Color('orange'),
        Color('cyan'),
        Color('yellow'),
        Color('black'),
        Color('brown'),
    ]

    def __init__(self, position, radius, color, max_pos, min_pos):
        self.position = position
        self._color = color
        self._max_pos = max_pos
        self._min_pos = min_pos
        self._radius = radius
        self.target_position = position

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


class PickAndPlaceEnv(MultitaskEnv, Serializable):
    """
    A simple env where a 'cursor' robot can push objects around.

    TODO: refactor to have base class shared with point2d.py code
    TODO: rather than recreating a drawer, just save the previous drawers
    """

    def __init__(
            self,
            num_objects=2,
            # Environment dynamics
            action_scale=1.0,
            ball_radius=.75,
            boundary_dist=4,
            object_radius=0.50,
            min_grab_distance=0.5,
            walls=None,
            # Rewards
            action_l2norm_penalty=0,
            reward_type="dense",
            success_threshold=0.60,
            # Reset settings
            fixed_goal=None,
            fixed_init_position=None,
            init_position_strategy='random',
            start_near_object=False,
            # Visualization settings
            images_are_rgb=True,
            render_dt_msec=0,
            render_onscreen=False,
            render_size=84,
            get_image_base_render_size=None,
            show_goal=True,
            # Goal sampling
            goal_samplers=None,
            goal_sampling_mode='random',
            num_presampled_goals=10000,

            object_reward_only=False,
    ):
        """
        :param get_image_base_render_size: If set, then the drawer will always
        generate images according to this size, and (smoothly) downsampled
        images will be returned by `get_image`
        """
        walls = walls or []
        if fixed_goal is not None:
            fixed_goal = np.array(fixed_goal)
        if action_scale <= 0:
            raise ValueError("Invalid action scale: {}".format(
                action_scale
            ))
        if init_position_strategy not in {'random', 'on_random_object', 'fixed'}:
            raise ValueError('Invalid init position strategy: {}'.format(
                init_position_strategy
            ))

        self.quick_init(locals())
        self.render_dt_msec = render_dt_msec
        self.action_l2norm_penalty = action_l2norm_penalty
        self.render_onscreen = render_onscreen
        self.render_size = render_size
        self.reward_type = reward_type
        self.action_scale = action_scale
        self.success_threshold = success_threshold
        self.boundary_dist = boundary_dist
        self.ball_radius = ball_radius
        self.walls = walls
        self.fixed_goal = fixed_goal
        self.init_position_strategy = init_position_strategy
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
        self.num_objects = num_objects
        self.obs_range = spaces.Box(-o, o, dtype='float32')
        self.observation_space = spaces.Dict([
            ('observation', self.obs_range),
            ('desired_goal', self.obs_range),
            ('achieved_goal', self.obs_range),
            ('state_observation', self.obs_range),
            ('state_desired_goal', self.obs_range),
            ('state_achieved_goal', self.obs_range),
        ])

        if get_image_base_render_size:
            base_width, base_height = get_image_base_render_size
            self._drawer = PygameViewer(
                screen_width=base_width,
                screen_height=base_height,
                x_bounds=(-self.boundary_dist - self.ball_radius, self.boundary_dist + self.ball_radius),
                y_bounds=(-self.boundary_dist - self.ball_radius, self.boundary_dist + self.ball_radius),
                render_onscreen=self.render_onscreen,
            )
            self._fixed_get_image_render_size = True
        else:
            self._drawer = None
            self._fixed_get_image_render_size = False
        self._render_drawer = None
        if fixed_init_position is None:
            fixed_init_position = np.zeros_like(self.obs_range.low)
        self._fixed_init_position = fixed_init_position

        self._presampled_goals = None
        goal_samplers = goal_samplers or {}
        goal_samplers['fixed'] = PickAndPlaceEnv._sample_fixed_goal
        goal_samplers['presampled'] = PickAndPlaceEnv._sample_presampled_goals
        goal_samplers['random'] = PickAndPlaceEnv._sample_random_feasible_goals
        self._custom_goal_sampler = goal_samplers
        self._num_presampled_goals = num_presampled_goals
        if goal_sampling_mode is None:
            if fixed_goal:
                goal_sampling_mode = 'fixed'
            else:
                goal_sampling_mode = 'presampled'
        self.goal_sampling_mode = goal_sampling_mode
        self.object_reward_only = object_reward_only

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
        is_success = distance_to_target < self.success_threshold
        info = {
            'distance_to_target_cursor': distance_to_target,
            'success_cursor': is_success,
        }
        for i, obj in enumerate(self.objects):
            obj_distance = obj.distance_to_target()
            success = obj_distance < self.success_threshold
            info['distance_to_target_obj_{}'.format(i)] = obj_distance
            info['success_obj_{}'.format(i)] = success
        for i, obj in enumerate(self.objects):
            obj_distance = np.linalg.norm(self.cursor.position - obj.position)
            success = obj_distance < self.success_threshold
            info['cursor_distance_obj_{}'.format(i)] = obj_distance
            info['cursor_success_obj_{}'.format(i)] = success
        return info

    def reset(self):
        goal = self.sample_goal()['state_desired_goal']
        self._set_target_positions(goal)

        if self.init_position_strategy == 'random':
            init_pos = (
                self.observation_space.spaces['state_observation'].sample()
            )
        elif self.init_position_strategy == 'fixed':
            init_pos = self._fixed_init_position.copy()
        elif self.init_position_strategy == 'on_random_object':
            init_pos = (
                self.observation_space.spaces['state_observation'].sample()
            )
            start_i = 2 + 2 * random.randint(0, len(self._all_objects) - 2)
            end_i = start_i + 2
            init_pos[:2] = init_pos[start_i:end_i].copy()
        else:
            raise ValueError(self.init_position_strategy)
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

        if self.object_reward_only:
            achieved_goals = achieved_goals[:, 2:]
            desired_goals = desired_goals[:, 2:]
        d = np.linalg.norm(achieved_goals - desired_goals, axis=-1)
        if self.reward_type == "sparse":
            return -(d > self.success_threshold * len(self._all_objects)
                     ).astype(np.float32)
        elif self.reward_type == "dense":
            return -d
        elif self.reward_type == "dense_l1":
            return -np.linalg.norm(achieved_goals - desired_goals, axis=-1,
                                   ord=1)
        elif self.reward_type == 'vectorized_dense':
            return -np.abs(achieved_goals - desired_goals)
        else:
            raise NotImplementedError()

    def get_goal(self):
        return {
            'desired_goal': self._get_target_positions(),
            'state_desired_goal': self._get_target_positions(),
        }

    def set_goal(self, goal):
        self._set_target_positions(goal['state_desired_goal'])

    def sample_goals(self, batch_size):
        goal_sampler = self._custom_goal_sampler[self.goal_sampling_mode]
        return goal_sampler(self, batch_size)

    def _sample_random_feasible_goals(self, batch_size):
        if len(self.walls) > 0:
            goals = np.zeros(
                (batch_size, self.obs_range.low.size)
            )
            for b in range(batch_size):
                goals[b, :] = self._sample_position(
                    self.obs_range.low,
                    self.obs_range.high,
                )
        else:
            goals = np.random.uniform(
                self.obs_range.low,
                self.obs_range.high,
                size=(batch_size, self.obs_range.low.size),
            )
        state_goals = goals
        return {
            'desired_goal': goals,
            'state_desired_goal': state_goals,
        }

    def _sample_fixed_goal(self, batch_size):
        goals = state_goals = np.repeat(
            self.fixed_goal.copy()[None],
            batch_size,
            0
        )
        return {
            'desired_goal': goals,
            'state_desired_goal': state_goals,
        }

    def _sample_presampled_goals(self, batch_size):
        if self._presampled_goals is None:
            self._presampled_goals = self._sample_random_feasible_goals(
                self._num_presampled_goals
            )
        random_idxs = np.random.choice(len(list(self._presampled_goals.values())[0]), size=batch_size)
        goals = self._presampled_goals['desired_goal'][random_idxs]
        state_goals = self._presampled_goals['state_desired_goal'][random_idxs]
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
        if self._drawer is None or (
            not self._fixed_get_image_render_size
            and (self._drawer.width != width or self._drawer.height != height)
        ):
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
        if width and height:
            wh_size = (width, height)
        else:
            wh_size = None
        img = self._drawer.get_image(wh_size)
        if self.images_are_rgb:
            return img.transpose((1, 0, 2))
        else:
            r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
            img = (r + g + b).transpose() / 3
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
        drawer.fill(Color('white')) #'black'
        for obj in self._all_objects:
            obj.draw(drawer, draw_target_position=self._show_goal)

        for wall in self.walls:
            draw_wall(drawer, wall)
        drawer.render()

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

    def goal_conditioned_diagnostics(self, paths, goals):
        statistics = OrderedDict()
        stat_to_lists = defaultdict(list)
        for path, goal in zip(paths, goals):
            difference = path['observations'] - goal
            num_obj_success = np.zeros(difference.shape[0])
            num_obj_1_2_success = np.zeros(difference.shape[0])
            for i in range(len(self._all_objects)):
                distance = np.linalg.norm(
                    difference[:, 2*i:2*i+2], axis=-1
                )
                distance_key = 'distance_to_target_obj_{}'.format(i)
                stat_to_lists[distance_key].append(distance)
                success_key = 'success_obj_{}'.format(i)
                obj_success = (distance < self.success_threshold)
                stat_to_lists[success_key].append(obj_success)
                num_obj_success += obj_success
                if i in [1, 2]:
                    num_obj_1_2_success += obj_success
            stat_to_lists['num_obj_success'].append(num_obj_success)
            stat_to_lists['num_obj_1_2_success'].append(num_obj_1_2_success)
        for stat_name, stat_list in stat_to_lists.items():
            statistics.update(create_stats_ordered_dict(
                'env_infos/{}'.format(stat_name),
                stat_list,
                always_show_all_stats=True,
                exclude_max_min=True,
            ))
            statistics.update(create_stats_ordered_dict(
                'env_infos/final/{}'.format(stat_name),
                [s[-1:] for s in stat_list],
                always_show_all_stats=True,
                exclude_max_min=True,
            ))
            statistics.update(create_stats_ordered_dict(
                'env_infos/initial/{}'.format(stat_name),
                [s[:1] for s in stat_list],
                always_show_all_stats=True,
                exclude_max_min=True,
            ))
        return statistics


    def get_image_v(
            self,
            policy,
            qf,
            get_state_func,
            get_goal_func,
            get_mask_func,
            obj_ids,
            imsize=None
    ):
        nx, ny = (50, 50)
        x = np.linspace(-4, 4, nx)
        y = np.linspace(-4, 4, ny)
        xv, yv = np.meshgrid(x, y)

        curr_state = get_state_func()
        curr_goal = get_goal_func()
        curr_mask = get_mask_func()

        if (curr_state is None) or (curr_goal is None) or (curr_mask is None):
            return [None] * len(obj_ids)

        sweep_goal = np.tile(curr_goal.reshape((1, -1)), (nx * ny, 1))
        sweep_mask = np.tile(curr_mask.reshape((1, -1)), (nx * ny, 1))

        list_of_v_vals = []

        ### sweep the state ###
        for obj_id in obj_ids:
            if obj_id is None:  ### infer the obj id from the mask
                indices = np.argwhere(curr_mask == 1)[-1]
                obj_id = indices[0] // 2

            start_idx = obj_id * 2
            end_idx = start_idx + 2
            sweep_state = np.tile(curr_state.reshape((1, -1)), (nx * ny, 1))
            sweep_state[:, start_idx:end_idx] = np.stack((xv, yv), axis=2).reshape((-1, 2))

            sweep_obs = np.hstack((sweep_state, sweep_goal, sweep_mask))
            sweep_actions = policy.get_actions(sweep_obs)

            from railrl.torch.core import torch_ify, np_ify
            v_vals = qf(
                torch_ify(sweep_obs),
                torch_ify(sweep_actions),
            )
            v_vals = np_ify(v_vals)
            v_vals = v_vals.reshape((nx, ny)) / 100
            list_of_v_vals.append(v_vals)

        list_of_images = []
        for (i, v_vals) in enumerate(list_of_v_vals):
            # vmin, vmax = None, None
            # vmin = np.percentile(list_of_v_vals, 90)
            # vmax = np.percentile(list_of_v_vals, 98)
            vmin = np.min(list_of_v_vals) + (np.max(list_of_v_vals) - np.min(list_of_v_vals)) * 0.70
            vmax = np.max(list_of_v_vals)
            # vmin, vmax = -50, 0

            fig, ax = self.draw_plt_objects(draw_state=True, draw_goal=True, imsize=imsize)
            ax.imshow(
                v_vals,
                extent=[-4, 4, -4, 4],
                cmap=plt.get_cmap(
                    # 'plasma'
                    'Greys'
                ),
                interpolation='nearest',
                vmax=vmax,
                vmin=vmin,
                origin='bottom',  # <-- Important! By default top left is (0, 0)
            )
            image = self.plt_to_numpy(fig)
            list_of_images.append(image)

        return list_of_images

    def get_image_pi(
            self,
            policy,
            get_state_func,
            get_goal_func,
            get_mask_func,
            obj_ids,
            imsize=None
    ):
        nx, ny = (15, 15)
        x = np.linspace(-4, 4, nx)
        y = np.linspace(-4, 4, ny)
        xv, yv = np.meshgrid(x, y)

        curr_state = get_state_func()
        curr_goal = get_goal_func()
        curr_mask = get_mask_func()

        if (curr_state is None) or (curr_goal is None) or (curr_mask is None):
            return [None] * len(obj_ids)

        sweep_goal = np.tile(curr_goal.reshape((1, -1)), (nx * ny, 1))
        sweep_mask = np.tile(curr_mask.reshape((1, -1)), (nx * ny, 1))

        list_of_actions = []

        for obj_id in obj_ids:
            if obj_id is None:  ### infer the obj id from the mask
                indices = np.argwhere(curr_mask == 1)[-1]
                obj_id = indices[0] // 2

            ### sweep the state ###
            start_idx = obj_id * 2
            end_idx = start_idx + 2
            sweep_state = np.tile(curr_state.reshape((1, -1)), (nx * ny, 1))
            sweep_state[:, start_idx:end_idx] = np.stack((xv, yv), axis=2).reshape((-1, 2))

            sweep_state[:, 0:2] = sweep_state[:, start_idx:end_idx]

            sweep_obs = np.hstack((sweep_state, sweep_goal, sweep_mask))
            sweep_actions = policy.get_actions(sweep_obs).reshape((nx, ny, -1))

            list_of_actions.append(sweep_actions)

        list_of_images = []
        for (i, actions) in enumerate(list_of_actions):
            fig, ax = self.draw_plt_objects(
                draw_state=True,
                draw_goal=True,
                imsize=imsize,
            )

            dx, dy = actions[:, :, 0], actions[:, :, 1]
            dx = np.clip(dx, a_min=-1, a_max=1)
            dy = np.clip(dy, a_min=-1, a_max=1)
            ax.quiver(
                xv, yv,
                dx, -1 * dy,
                scale=2.5e1,
                headwidth=5,
            )

            image = self.plt_to_numpy(fig)
            list_of_images.append(image)

        return list_of_images

    def draw_plt_objects(self, extent=[-4, 4, -4, 4], draw_state=True, draw_goal=True, imsize=None):
        fig, ax = plt.subplots()
        ax.set_ylim(extent[2:4])
        ax.set_xlim(extent[0:2])
        ax.set_ylim(ax.get_ylim()[::-1])
        DPI = fig.get_dpi()
        if imsize is None:
            fig.set_size_inches(self.render_size / float(DPI), self.render_size / float(DPI))
        else:
            fig.set_size_inches(imsize / float(DPI), imsize / float(DPI))

        if draw_state:
            positions = self._get_positions()
            for i in range(self.num_objects + 1):
                ball = plt.Circle(
                    positions[i*2:i*2+2],
                    self.ball_radius * 0.40,
                    color=Object.IDX_TO_COLOR[i].normalize(),
                    fill=True,
                )
                ax.add_artist(ball)
        if draw_goal:
            target_positions = self._get_target_positions()
            for i in range(1, self.num_objects + 1):
                ball = plt.Circle(
                    target_positions[i*2:i*2+2],
                    self.ball_radius * 0.60,
                    color=Object.IDX_TO_COLOR[i].normalize(),
                    fill=False,
                )
                ax.add_artist(ball)

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        fig.subplots_adjust(bottom=0)
        fig.subplots_adjust(top=1)
        fig.subplots_adjust(right=1)
        fig.subplots_adjust(left=0)
        ax.axis('off')

        return fig, ax

    def plt_to_numpy(self, fig):
        fig.canvas.draw()
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()
        return data

class PickAndPlace1DEnv(PickAndPlaceEnv):
    def __init__(self, *args, **kwargs):
        self.quick_init(locals())
        super().__init__(*args, **kwargs)

        o = self.boundary_dist * np.ones(2 * (self.num_objects+1))
        for obj_idx in range(self.num_objects + 1):
            o[obj_idx * 2] = 0.0
        self.obs_range = spaces.Box(-o, o, dtype='float32')
        self.observation_space = spaces.Dict([
            ('observation', self.obs_range),
            ('desired_goal', self.obs_range),
            ('achieved_goal', self.obs_range),
            ('state_observation', self.obs_range),
            ('state_desired_goal', self.obs_range),
            ('state_achieved_goal', self.obs_range),
        ])

    def step(self, raw_action):
        action = raw_action.copy()
        action[0] = 0.0
        return super().step(action)
