import abc
import copy
import numpy as np
from gym.spaces import Box, Dict

from multiworld.core.serializable import Serializable
from multiworld.envs.env_util import concatenate_box_spaces
from multiworld.envs.mujoco.mujoco_env import MujocoEnv
from multiworld.core.multitask_env import MultitaskEnv
from multiworld.envs.env_util import get_asset_full_path

from collections import OrderedDict
from multiworld.envs.env_util import (
    get_stat_in_paths,
    create_stats_ordered_dict,
)


class WheeledCarEnv(MujocoEnv, Serializable, MultitaskEnv): #, metaclass=abc.ABCMeta):
    def __init__(
            self,
            reward_type='state_distance',
            norm_order=2,
            action_scale=20,
            frame_skip=3,
            two_frames=False,
            vel_in_state=True,
            z_in_state=True,
            car_low=(-1.90, -1.90), #list([-1.60, -1.60]),
            car_high=(1.90, 1.90), #list([1.60, 1.60]),
            goal_low=(-1.90, -1.90), #list([-1.60, -1.60]),
            goal_high=(1.90, 1.90), #list([1.60, 1.60]),
            model_path='wheeled_car.xml',
            reset_low=None,
            reset_high=None,
            *args,
            **kwargs
    ):
        self.quick_init(locals())
        MultitaskEnv.__init__(self)
        MujocoEnv.__init__(self,
                           model_path=get_asset_full_path('locomotion/' + model_path),
                           frame_skip=frame_skip,
                           **kwargs)

        self.action_space = Box(np.array([-1, -1]), np.array([1, 1]), dtype=np.float32)
        self.action_scale = action_scale

        # set radius
        if model_path in ['wheeled_car.xml', 'wheeled_car_u_wall.xml']:
            self.car_radius = np.sqrt(2) * 0.34
        elif model_path == 'wheeled_car_old.xml':
            self.car_radius = 0.30
        else:
            raise NotImplementedError

        # set walls
        if model_path == 'wheeled_car_u_wall.xml':
            self.walls = [
                Wall(0, -0.5, 0.75, 0.05, self.car_radius),
                Wall(0.70, 0.05, 0.05, 0.5, self.car_radius),
                Wall(-0.70, 0.05, 0.05, 0.5, self.car_radius),
            ]
        else:
            self.walls = []

        self.reward_type = reward_type
        self.norm_order = norm_order

        self.car_low, self.car_high = np.array(car_low), np.array(car_high)
        self.goal_low, self.goal_high = np.array(goal_low), np.array(goal_high)

        if reset_low is None:
            self.reset_low = np.array(car_low)
        else:
            self.reset_low = np.array(reset_low)
        if reset_high is None:
            self.reset_high = np.array(car_high)
        else:
            self.reset_high = np.array(reset_high)

        self.car_low += self.car_radius
        self.car_high -= self.car_radius
        self.goal_low += self.car_radius
        self.goal_high -= self.car_radius
        self.reset_low += self.car_radius
        self.reset_high -= self.car_radius

        self.two_frames = two_frames
        self.vel_in_state = vel_in_state
        self.z_in_state = z_in_state

        # x and y pos
        obs_space_low = np.copy(self.car_low)
        obs_space_high = np.copy(self.car_high)
        goal_space_low = np.copy(self.goal_low)
        goal_space_high = np.copy(self.goal_high)

        # z pos
        if self.z_in_state:
            obs_space_low = np.concatenate((obs_space_low, [-1]))
            obs_space_high = np.concatenate((obs_space_high, [0.03]))
            goal_space_low = np.concatenate((goal_space_low, [0]))
            goal_space_high = np.concatenate((goal_space_high, [0]))

        # sin and cos
        obs_space_low = np.concatenate((obs_space_low, [-1, -1]))
        obs_space_high = np.concatenate((obs_space_high, [1, 1]))
        goal_space_low = np.concatenate((goal_space_low, [-1, -1]))
        goal_space_high = np.concatenate((goal_space_high, [1, 1]))

        if self.vel_in_state:
            # x and y vel
            obs_space_low = np.concatenate((obs_space_low, [-10, -10]))
            obs_space_high = np.concatenate((obs_space_high, [10, 10]))
            goal_space_low = np.concatenate((goal_space_low, [0, 0]))
            goal_space_high = np.concatenate((goal_space_high, [0, 0]))

            # z vel
            if self.z_in_state:
                obs_space_low = np.concatenate((obs_space_low, [-10]))
                obs_space_high = np.concatenate((obs_space_high, [10]))
                goal_space_low = np.concatenate((goal_space_low, [0]))
                goal_space_high = np.concatenate((goal_space_high, [0]))

            # theta vel
            obs_space_low = np.concatenate((obs_space_low, [-10]))
            obs_space_high = np.concatenate((obs_space_high, [10]))
            goal_space_low = np.concatenate((goal_space_low, [0]))
            goal_space_high = np.concatenate((goal_space_high, [0]))

        self.obs_space = Box(obs_space_low, obs_space_high, dtype=np.float32)
        self.goal_space = Box(goal_space_low, goal_space_high, dtype=np.float32)
        if self.two_frames:
            self.obs_space = concatenate_box_spaces(self.obs_space, self.obs_space)
            self.goal_space = concatenate_box_spaces(self.goal_space, self.goal_space)

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

        self._state_goal = None
        self._cur_obs = None
        self._prev_obs = None
        self.reset()

    def step(self, action):
        self._prev_obs = self._cur_obs
        action = self.action_scale * action
        self.do_simulation(np.array(action), self.frame_skip)
        ob = self._get_obs()
        reward = self.compute_reward(action, ob)
        state, goal = ob['state_observation'], ob['state_desired_goal']
        state_diff = np.linalg.norm(state - goal)
        if self.z_in_state:
            pos_diff = np.linalg.norm(state[:3] - goal[:3])
            angle_state, angle_goal = np.arctan2(state[3], state[4]), np.arctan2(goal[3], goal[4])
            angle_diff = np.abs(np.arctan2(np.sin(angle_state - angle_goal), np.cos(angle_state - angle_goal)))
            pos_angle_diff = np.linalg.norm(state[:5] - goal[:5])
        else:
            pos_diff = np.linalg.norm(state[:2] - goal[:2])
            angle_state, angle_goal = np.arctan2(state[2], state[3]), np.arctan2(goal[2], goal[3])
            angle_diff = np.abs(np.arctan2(np.sin(angle_state - angle_goal), np.cos(angle_state - angle_goal)))
            pos_angle_diff = np.linalg.norm(state[:4] - goal[:4])

        pos_success = float(pos_diff < 0.6),
        angle_success = float(angle_diff < 0.30),

        info = {
            'state_diff': state_diff,
            'pos_diff': pos_diff,
            'angle_diff': angle_diff,
            'pos_angle_diff': pos_angle_diff,
            'pos_success': pos_success,
            'angle_success': angle_success,
        }
        if self.vel_in_state:
            if self.z_in_state:
                info['velocity_diff'] = np.linalg.norm(state[-4:-1] - goal[-4:-1])
            else:
                info['velocity_diff'] = np.linalg.norm(state[-3:-1] - goal[-3:-1])
            info['angular_velocity_diff'] = np.linalg.norm(state[-1] - goal[-1])
        done = False
        return ob, reward, done, info

    def _get_obs(self):
        qpos = list(self.sim.data.qpos.flat)
        if self.z_in_state:
            flat_obs = qpos[:3]
        else:
            flat_obs = qpos[:2]
        flat_obs = flat_obs + [np.sin(qpos[-3]), np.cos(qpos[-3])]
        if self.vel_in_state:
            qvel = list(self.sim.data.qvel.flat)
            flat_obs = flat_obs + qvel[:2]
            if self.z_in_state:
                flat_obs = flat_obs + [qvel[2]]
            flat_obs = flat_obs + [qvel[3]]

        flat_obs = np.array(flat_obs)

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

        if self.two_frames:
            if self._prev_obs is None:
                self._prev_obs = self._cur_obs
            frames = self.merge_frames(self._prev_obs, self._cur_obs)
            return frames

        return self._cur_obs

    def merge_frames(self, frame1, frame2):
        frame = {}
        for key in frame1.keys():
            frame[key] = np.concatenate((frame1[key], frame2[key]))
        return frame

    def get_goal(self):
        if self.two_frames:
            return {
                'desired_goal': np.concatenate((self._state_goal, self._state_goal)),
                'state_desired_goal': np.concatenate((self._state_goal, self._state_goal)),
            }
        else:
            return {
                'desired_goal': self._state_goal,
                'state_desired_goal': self._state_goal,
            }

    def sample_goals(self, batch_size):
        goals = np.random.uniform(
            self.goal_space.low,
            self.goal_space.high,
            size=(batch_size, self.goal_space.low.size),
        )
        if self.two_frames:
            goals = goals[:,:int(self.goal_space.low.size/2)]

        angles = np.random.uniform(0, 2 * np.pi, batch_size)
        if self.z_in_state:
            goals[:, 3], goals[:, 4] = np.sin(angles), np.cos(angles)
        else:
            goals[:, 2], goals[:, 3] = np.sin(angles), np.cos(angles)

        # count = 0
        # for goal in goals:
        #     while self._position_inside_wall(goal[:2]):
        #         count += 1
        #         goal[:2] = np.random.uniform(self.goal_space.low[:2], self.goal_space.high[:2])
        # print(count)

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

        if self.two_frames:
            return {
                'desired_goal': np.concatenate((goals, goals), axis=1),
                'state_desired_goal': np.concatenate((goals, goals), axis=1),
            }
        else:
            return {
                'desired_goal': goals,
                'state_desired_goal': goals,
            }

    def generate_expert_subgoals(self, num_subgoals):
        ob_and_goal = self._get_obs()
        ob = ob_and_goal['state_observation']
        goal = ob_and_goal['state_desired_goal']

        subgoals = []

        theta = np.pi / 2
        s1 = np.array([1.3, -1.0, np.sin(theta), np.cos(theta)])
        s2 = np.array([1.3, 1.0, np.sin(theta), np.cos(theta)])
        s3 = np.array([0.0, 1.0, np.sin(theta), np.cos(theta)])
        subgoals += [s1, s2, s3]
        # subgoals += [s1, s1, s1, s1]

        if self.two_frames:
            for i in range(len(subgoals)):
                subgoals[i] = np.tile(subgoals[i], 2)
        subgoals.append(goal)

        if len(subgoals) == 0:
            subgoals = np.tile(goal, num_subgoals).reshape(-1, len(goal))

        return np.array(subgoals)

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
        if self.reward_type == 'vectorized_state_distance':
            r = -np.abs(diff)
        elif self.reward_type == 'state_distance':
            r = -np.linalg.norm(diff, ord=self.norm_order, axis=1)
        elif self.reward_type == 'pos_distance':
            if self.two_frames:
                goal_dim = int(diff.shape[1]/2)
                diff1, diff2 = diff[:,:goal_dim], diff[:,-goal_dim:]
                if self.z_in_state:
                    diff = np.hstack((diff1[:,:3], diff2[:,:3]))
                else:
                    diff = np.hstack((diff1[:, :2], diff2[:, :2]))
            else:
                if self.z_in_state:
                    diff = diff[:,:3]
                else:
                    diff = diff[:, :2]
            r = -np.linalg.norm(diff, ord=self.norm_order, axis=1)
        elif self.reward_type == 'pos_angle_distance':
            if self.two_frames:
                goal_dim = int(diff.shape[1] / 2)
                diff1, diff2 = diff[:, :goal_dim], diff[:, -goal_dim:]
                if self.z_in_state:
                    diff = np.hstack((diff1[:,:5], diff2[:,:5]))
                else:
                    diff = np.hstack((diff1[:, :4], diff2[:, :4]))
            else:
                if self.z_in_state:
                    diff = diff[:,:5]
                else:
                    diff = diff[:, :4]
            r = -np.linalg.norm(diff, ord=self.norm_order, axis=1)
        else:
            raise NotImplementedError("Invalid/no reward type.")
        return r

    def reset_model(self):
        self._reset_car()
        self.set_goal(self.sample_goal())
        self.sim.forward()
        self._prev_obs = None
        self._cur_obs = None
        return self._get_obs()

    def _reset_car(self):
        qpos = np.zeros(6)
        qvel = np.zeros(6)
        pos_2d = np.random.uniform(self.reset_low, self.reset_high)
        while self._position_inside_wall(pos_2d):
            pos_2d = np.random.uniform(self.reset_low, self.reset_high)
        qpos[0:2] = pos_2d
        qpos[3] = np.random.uniform(0, 2 * np.pi)
        self.set_state(qpos, qvel)

    def set_goal(self, goal):
        if self.two_frames:
            self._state_goal = goal['state_desired_goal'][int(len(goal['state_desired_goal'])/2):]
        else:
            self._state_goal = goal['state_desired_goal']
        self._prev_obs = None
        self._cur_obs = None

    def set_to_goal(self, goal):
        state_goal = goal['state_desired_goal']
        if self.two_frames:
            goal_dim = len(self.goal_space.low) // 2
            state_goal = state_goal[:goal_dim]
        qpos, qvel = np.zeros(6), np.zeros(6)
        if self.z_in_state:
            qpos[0:3] = state_goal[0:3] #xyz pos
            qpos[3] = np.arctan2(state_goal[3], state_goal[4]) #angle
        else:
            qpos[0:2] = state_goal[0:2] #xyz pos
            qpos[3] = np.arctan2(state_goal[2], state_goal[3]) #angle

        if self.vel_in_state:
            if self.z_in_state:
                qvel[0:3] = state_goal[-4:-1] #vel_{xyz}
            else:
                qvel[0:2] = state_goal[-3:-1]  # vel_{xy}
            qvel[3] = state_goal[-1] #vel_angle
        self.set_state(qpos, qvel)
        self._prev_obs = None
        self._cur_obs = None

    def get_env_state(self):
        joint_state = self.sim.get_state()
        state = joint_state, self._state_goal, self._cur_obs, self._prev_obs
        return copy.deepcopy(state)

    def set_env_state(self, state):
        state, goal, cur_obs, prev_obs = state
        self.sim.set_state(state)
        self.sim.forward()
        self._state_goal = goal
        self._cur_obs = cur_obs
        self._prev_obs = prev_obs

    def valid_state(self, state):
        return self.valid_states(state[None])[0]

    def valid_states(self, states):
        if self.z_in_state:
            states[:,3] = np.clip(states[:,3], -1, 1) #sin
            states[:,4] = np.clip(states[:,4], -1, 1) #cos
            angles = np.arcsin(states[:, 3])
        else:
            states[:,2] = np.clip(states[:,2], -1, 1) #sin
            states[:,3] = np.clip(states[:,3], -1, 1) #cos
            angles = np.arcsin(states[:, 2])
        for i in range(len(angles)):
            if self.z_in_state and states[i][4] <= 0:
                angles[i] = np.pi - angles[i]
            elif not self.z_in_state and states[i][3] <= 0:
                angles[i] = np.pi - angles[i]
        if self.z_in_state:
            states[:,3], states[:,4] = np.sin(angles), np.cos(angles)
        else:
            states[:, 2], states[:, 3] = np.sin(angles), np.cos(angles)
        return states

    def get_diagnostics(self, paths, prefix=''):
        statistics = OrderedDict()
        list_of_stat_names = [
            'state_diff',
            'pos_diff',
            'angle_diff',
            'pos_angle_diff',
            'pos_success',
            'angle_success',
        ]
        if self.vel_in_state:
            list_of_stat_names.append('velocity_diff')
            list_of_stat_names.append('angular_velocity_diff')

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
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.lookat[0] = 0.0
        self.viewer.cam.lookat[1] = 0.0
        self.viewer.cam.lookat[2] = 0.5
        self.viewer.cam.distance = 6.5
        self.viewer.cam.elevation = -90


class Wall(object, metaclass=abc.ABCMeta):
    def __init__(self, x_center, y_center, x_thickness, y_thickness, min_dist):
        self.min_x = x_center - x_thickness - min_dist
        self.max_x = x_center + x_thickness + min_dist
        self.min_y = y_center - y_thickness - min_dist
        self.max_y = y_center + y_thickness + min_dist

    def contains_point(self, point):
        return (self.min_x < point[0] < self.max_x) and (self.min_y < point[1] < self.max_y)

    def contains_points(self, points):
        return (self.min_x < points[:,0]) * (points[:,0] < self.max_x) \
               * (self.min_y < points[:,1]) * (points[:,1] < self.max_y)