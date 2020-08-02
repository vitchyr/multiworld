import roboverse.bullet as bullet
import numpy as np
import pybullet as p
from gym.spaces import Box, Dict
from collections import OrderedDict
from roboverse.envs.sawyer_base import SawyerBaseEnv
from roboverse.bullet.misc import load_obj, deg_to_quat, quat_to_deg
from bullet_objects import loader, metadata
import os.path as osp
import importlib.util
import random
import pickle
import gym

easy_objects = [bullet.objects.lego, bullet.objects.duck, bullet.objects.cube,
    bullet.objects.spam, bullet.objects.lid]
test_set = ['mug', 'square_deep_bowl', 'bathtub', 'crooked_lid_trash_can', 'beer_bottle', 
    'l_automatic_faucet', 'toilet_bowl', 'narrow_top_vase']

class SawyerDistractorReachingV0(SawyerBaseEnv):

    def __init__(self,
                 goal_pos=(0.75, 0.2, -0.1),
                 obs_img_dim=48,
                 success_threshold=0.05,
                 transpose_image=False,
                 invisible_robot=False,
                 object_subset='easy',
                 task='gr_hand',
                 pickup_eps=-0.35,
                 num_objects=5,
                 DoF=3,
                 *args,
                 **kwargs
                 ):
        """
        Grasping env with a single object
        :param goal_pos: xyz coordinate of desired goal
        :param reward_type: one of 'shaped', 'sparse'
        :param reward_min: minimum possible reward per timestep
        :param randomize: whether to randomize the object position or not
        :param observation_mode: state, pixels, pixels_debug
        :param obs_img_dim: image dimensions for the observations
        :param transpose_image: first dimension is channel when true
        :param invisible_robot: the robot arm is invisible when set to True
        """
        assert DoF in [3, 4, 6]
        assert task in ['goal_reaching', 'pickup', 'gr_hand']
        assert object_subset in ['test', 'train', 'easy', '']
        assert num_objects <= 5
        print("Task Type: " + task)
        self.goal_pos = np.asarray(goal_pos)
        self.pickup_eps = pickup_eps
        self._observation_mode = 'state'
        self._transpose_image = transpose_image
        self._invisible_robot = invisible_robot
        self.image_shape = (obs_img_dim, obs_img_dim)
        self.image_length = np.prod(self.image_shape) * 3  # image has 3 channels
        self.object_subset = object_subset
        self.num_objects = num_objects
        self._ddeg_scale = 5
        self.task = task
        self.DoF = DoF
        print("HARD CODED GRIPPER TO BE -1")

        self.object_dict, self.scaling = self.get_object_info()
        self.curr_object = None

        self._object_position_low = (.62, -0.15, -.3)
        self._object_position_high = (.78, 0.15, -.3)

        if task == 'gr_hand':
            self.start_rew_ind = 0
        else:
            self.start_rew_ind = 4 if (self.DoF == 3) else 8

        self._success_threshold = success_threshold
        self.default_theta = bullet.deg_to_quat([180, 0, 0])
        self.obs_img_dim = obs_img_dim #+.15
        self._view_matrix_obs = bullet.get_view_matrix(
            target_pos=[.7, 0, -0.3], distance=0.3,
            yaw=90, pitch=-15, roll=0, up_axis_index=2)
        self._projection_matrix_obs = bullet.get_projection_matrix(
            self.obs_img_dim, self.obs_img_dim)
        self.dt = 0.1
        super().__init__(*args, **kwargs)

    def get_object_info(self):
        complete_object_dict, scaling = metadata.obj_path_map, metadata.path_scaling_map
        complete = self.object_subset is None
        train = self.object_subset == 'train'
        test = self.object_subset == 'test'

        object_dict = {}
        for k in complete_object_dict.keys():
            in_test = (k in test_set)
            if complete:
                object_dict[k] = complete_object_dict[k]
            if train and not in_test:
                object_dict[k] = complete_object_dict[k]
            if test and in_test:
                object_dict[k] = complete_object_dict[k]
        return object_dict, scaling


    def _set_spaces(self):
        act_dim = self.DoF + 1
        act_bound = 1
        act_high = np.ones(act_dim) * act_bound
        self.action_space = gym.spaces.Box(-act_high, act_high)

        observation_dim = 4 + 7 * self.num_objects
        if self.DoF > 3:
            # Add wrist theta
            observation_dim += 4

        obs_bound = 100
        obs_high = np.ones(observation_dim) * obs_bound
        state_space = gym.spaces.Box(-obs_high, obs_high)

        self.observation_space = Dict([
            ('observation', state_space),
            ('state_observation', state_space),
            ('desired_goal', state_space),
            ('state_desired_goal', state_space),
            ('achieved_goal', state_space),
            ('state_achieved_goal', state_space),
        ])

    def _load_table(self):
        if self._invisible_robot:
            self._sawyer = bullet.objects.sawyer_invisible()
        else:
            self._sawyer = bullet.objects.sawyer_hand_visual_only()
        self._table = bullet.objects.table()
        self._objects = {}
        self._sensors = {}
        self._workspace = bullet.Sensor(self._sawyer,
            xyz_min=self._pos_low, xyz_max=self._pos_high,
            visualize=False, rgba=[0,1,0,.1])
        self._end_effector = bullet.get_index_by_attribute(
            self._sawyer, 'link_name', 'gripper_site')
        
        self._objects = {}

    def _load_meshes(self):
        self._objects = {}
        for i in range(self.num_objects):
            self.add_object_i(i)

    def _set_positions(self, pos):
        bullet.reset()
        bullet.setup_headless(self._timestep, solver_iterations=self._solver_iterations)
        self._load_table()

        start_ind = 4
        for i in range(self.num_objects):
            object_pos = pos[start_ind:start_ind + 3]
            object_quat = pos[start_ind + 3:start_ind + 7]
            start_ind += 7
            self.add_object_i(i, object_position=object_pos, quat=list(object_quat))

        self._format_state_query()
        hand_pos, gripper = pos[:3], pos[3]
        self._prev_pos = np.array(hand_pos)

        bullet.position_control(self._sawyer, self._end_effector, self._prev_pos, self.default_theta)

        action = np.array([0 for i in range(self.DoF)] + [gripper])
        for _ in range(10):
            self.step(action)

    # def _set_positions(self, pos):
    #     bullet.reset()
    #     bullet.setup_headless(self._timestep, solver_iterations=self._solver_iterations)
    #     self._load_table()

    #     for i in range(self.num_objects):
    #         start_ind = (i + 1) * 7 + 1
    #         object_pos = pos[start_ind:start_ind + 3]
    #         #object_pos[2] += 0.05 #Avoid collisions
    #         object_quat = pos[start_ind + 3:start_ind + 7]
    #         self.add_object_i(i, object_position=object_pos, quat=list(object_quat))

    #     self._format_state_query()
    #     hand_pos, hand_theta, gripper = pos[:3], pos[3:7], pos[7]
    #     self._prev_pos = np.array(hand_pos)

    #     low, high = self._pos_low[:], self._pos_high[:]
    #     bullet.position_control(self._sawyer, self._end_effector, self._prev_pos, hand_theta)

    #     action = np.array([0 for i in range(self.DoF)] + [gripper])
    #     for _ in range(10):
    #         self.step(action)

    def add_object_i(self, i, object_position=None, quat=[0, 0, 0, 1]):
        # Generate object random position
        if object_position is None:
            object_position = np.random.uniform(
                low=self._object_position_low,
                high=self._object_position_high
            )
        
        # Spawn object above table
        self._objects[i] = easy_objects[i](pos=object_position, quat=quat)

        # Allow the objects to land softly in low gravity
        p.setGravity(0, 0, -1)
        for _ in range(100):
            bullet.step()
        # After landing, bring to stop
        p.setGravity(0, 0, -10)
        for _ in range(100):
            bullet.step()

    def _format_action(self, *action):
        if self.DoF == 3:
            if len(action) == 1:
                delta_pos, gripper = action[0][:-1], action[0][-1]
            elif len(action) == 2:
                delta_pos, gripper = action[0], action[1]
            else:
                raise RuntimeError('Unrecognized action: {}'.format(action))
            return np.array(delta_pos), gripper
        elif self.DoF == 4:
            if len(action) == 1:
                delta_pos, delta_yaw, gripper = action[0][:3], action[0][3:4], action[0][-1]
            elif len(action) == 3:
                delta_pos, delta_yaw, gripper = action[0], action[1], action[2]
            else:
                raise RuntimeError('Unrecognized action: {}'.format(action))
            
            delta_angle = [0, 0, delta_yaw[0]]
            return np.array(delta_pos), np.array(delta_angle), gripper
        else:
            if len(action) == 1:
                delta_pos, delta_angle, gripper = action[0][:3], action[0][3:6], action[0][-1]
            elif len(action) == 3:
                delta_pos, delta_angle, gripper = action[0], action[1], action[2]
            else:
                raise RuntimeError('Unrecognized action: {}'.format(action))
            return np.array(delta_pos), np.array(delta_angle), gripper


    def step(self, *action):
        # Get positional information
        pos = bullet.get_link_state(self._sawyer, self._end_effector, 'pos')
        curr_angle = bullet.get_link_state(self._sawyer, self._end_effector, 'theta')
        default_angle = quat_to_deg(self.default_theta)
    
        # Keep necesary degrees of theta fixed
        if self.DoF == 3:
            angle = default_angle
        elif self.DoF == 4:
            angle = np.append(default_angle[:2], [curr_angle[2]])
        else:
            angle = curr_angle

        # If angle is part of action, use it
        if self.DoF == 3:
            delta_pos, gripper = self._format_action(*action)
        else:
            delta_pos, delta_angle, gripper = self._format_action(*action)
            angle += delta_angle * self._ddeg_scale

        # Update position and theta
        pos += delta_pos * self._action_scale
        pos = np.clip(pos, self._pos_low, self._pos_high)
        theta = deg_to_quat(angle)

        gripper = -1
        self._simulate(pos, theta, gripper)

        # Get tuple information
        observation = self.get_observation()
        info = self.get_info()
        reward = self.get_reward(info)
        done = False
        return observation, reward, done, info

    def get_info(self):
        end_effector_pos = self.get_end_effector_pos()
        gripper_goal_distance = np.linalg.norm(
            self.goal_pos - end_effector_pos)
        gripper_goal_success = int(gripper_goal_distance < self._success_threshold)

        info = {
            'gripper_goal_success': gripper_goal_success,
        }

        return info

    def get_contextual_diagnostics(self, paths, contexts):
        from multiworld.envs.env_util import create_stats_ordered_dict
        diagnostics = OrderedDict()
        state_key = "state_observation"
        goal_key = "state_desired_goal"
        values = []
        for i in range(len(paths)):
            state = paths[i]["observations"][-1][state_key][self.start_rew_ind:self.start_rew_ind + 3]
            height = state[2]
            goal = contexts[i][goal_key][self.start_rew_ind:self.start_rew_ind + 3]
            distance = np.linalg.norm(state - goal)
            values.append(distance)
            #values.append(height)
        diagnostics_key = goal_key + "/final/distance"
        #diagnostics_key = goal_key + "/final/height"
        diagnostics.update(create_stats_ordered_dict(
            diagnostics_key,
            values,
        ))

        values = []
        for i in range(len(paths)):
            for j in range(len(paths[i]["observations"])):
                state = paths[i]["observations"][j][state_key][self.start_rew_ind:self.start_rew_ind + 3]
                height = state[2]
                goal = contexts[i][goal_key][self.start_rew_ind:self.start_rew_ind + 3]
                distance = np.linalg.norm(state - goal)
                values.append(distance)
                #values.append(height)
        diagnostics_key = goal_key + "/distance"
        #diagnostics_key = goal_key + "/height"
        diagnostics.update(create_stats_ordered_dict(
            diagnostics_key,
            values,
        ))
        return diagnostics

    def render_obs(self):
        img, depth, segmentation = bullet.render(
            self.obs_img_dim, self.obs_img_dim, self._view_matrix_obs,
            self._projection_matrix_obs, shadow=0, gaussian_width=0)
        if self._transpose_image:
            img = np.transpose(img, (2, 0, 1))
        return img

    def set_goal(self, goal):
        self.goal_pos = goal['state_desired_goal'][self.start_rew_ind:self.start_rew_ind + 3]

    def get_image(self, width, height):
        image = np.float32(self.render_obs())
        return image

    def get_reward(self, info):
        if self.task == 'goal_reaching':
            return info['object_goal_success'] - 1
        elif self.task == 'gr_hand':
            return info['gripper_goal_success'] - 1
        elif self.task == 'pickup':
            return info['picked_up'] - 1

    def reset(self):
        # Load Enviorment
        bullet.reset()
        bullet.setup_headless(self._timestep, solver_iterations=self._solver_iterations)
        self._load_table()
        self._load_meshes()
        self._format_state_query()

        # Sample and load starting positions
        low, high = self._pos_low[:], self._pos_high[:]
        init_pos = np.random.uniform(low=low, high=high) #np.array(self._pos_init)
        self.goal_pos = np.random.uniform(low=low, high=high)
        bullet.position_control(self._sawyer, self._end_effector, init_pos, self.default_theta)

        # Move to starting positions
        action = np.array([0 for i in range(self.DoF)] + [-1])
        for _ in range(3):
            self.step(action)
        return self.get_observation()

    def format_obs(self, obs):
        if len(obs.shape) == 1:
            return obs.reshape(1, -1)
        return obs

    def compute_reward_pu(self, obs, actions, next_obs, contexts):
        obj_state = self.format_obs(next_obs['state_observation'])[:, self.start_rew_ind:self.start_rew_ind + 3]
        height = obj_state[:, 2]
        reward = (height > self.pickup_eps) - 1
        return reward
    
    def compute_reward_gr(self, obs, actions, next_obs, contexts):
        obj_state = self.format_obs(next_obs['state_observation'])[:, self.start_rew_ind:self.start_rew_ind + 3]
        obj_goal = self.format_obs(contexts['state_desired_goal'])[:, self.start_rew_ind:self.start_rew_ind + 3]
        object_goal_distance = np.linalg.norm(obj_state - obj_goal, axis=1)
        object_goal_success = object_goal_distance < self._success_threshold
        return object_goal_success - 1

    def compute_reward(self, obs, actions, next_obs, contexts):
        if self.task == 'goal_reaching' or self.task == 'gr_hand':
            return self.compute_reward_gr(obs, actions, next_obs, contexts)
        elif self.task == 'pickup':
            return self.compute_reward_pu(obs, actions, next_obs, contexts)

    def get_object_state(self, i):
        object_info = bullet.get_body_info(self._objects[i],
                                           quat_to_deg=False)
        object_pos = object_info['pos']
        object_theta = object_info['theta']
        return np.concatenate((object_pos, object_theta))

    def get_observation(self):
        left_tip_pos = bullet.get_link_state(
            self._sawyer, 'right_gripper_l_finger_joint', keys='pos')
        right_tip_pos = bullet.get_link_state(
            self._sawyer, 'right_gripper_r_finger_joint', keys='pos')
        left_tip_pos = np.asarray(left_tip_pos)
        right_tip_pos = np.asarray(right_tip_pos)
        hand_theta = bullet.get_link_state(self._sawyer, self._end_effector,
            'theta', quat_to_deg=False)

        gripper_tips_distance = [np.linalg.norm(
            left_tip_pos - right_tip_pos)]
        end_effector_pos = self.get_end_effector_pos()

        # Note: Fill in pseudo values for rest of goal pos to make it same shape as observation
        if self.DoF > 3:
            observation = np.concatenate((end_effector_pos, hand_theta, gripper_tips_distance))
            goal_pos = np.concatenate((self.goal_pos, hand_theta, gripper_tips_distance))
        else:
            observation = np.concatenate((end_effector_pos, gripper_tips_distance))
            goal_pos = np.concatenate((self.goal_pos, gripper_tips_distance))

        for i in range(self.num_objects):
            obj_info = self.get_object_state(i)
            observation = np.append(observation, obj_info)
            goal_pos = np.append(goal_pos, obj_info)

        obs_dict = dict(
            observation=observation,
            state_observation=observation,
            desired_goal=goal_pos,
            state_desired_goal=goal_pos,
            achieved_goal=observation,
            state_achieved_goal=observation,
            )

        return obs_dict