import roboverse.bullet as bullet
import numpy as np
from gym.spaces import Box, Dict
from collections import OrderedDict
from roboverse.envs.sawyer_base import SawyerBaseEnv
from multiworld.envs.env_util import create_stats_ordered_dict
import gym

class SawyerRigGRV0Env(SawyerBaseEnv):

    def __init__(self,
                 goal_pos=(0.75, 0.2, -0.1),
                 reward_type='shaped',
                 reward_min=-2.5,
                 randomize=True,
                 observation_mode='state',
                 obs_img_dim=48,
                 success_threshold=0.05,
                 transpose_image=False,
                 invisible_robot=False,
                 task='pickup',
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
        assert task in ['goal_reaching', 'pickup']
        print("Task Type: " + task)
        self.goal_pos = np.asarray(goal_pos)
        self._reward_type = reward_type
        self._reward_min = reward_min
        self._randomize = randomize
        self._observation_mode = observation_mode
        self._transpose_image = transpose_image
        self._invisible_robot = invisible_robot
        self.image_shape = (obs_img_dim, obs_img_dim)
        self.image_length = np.prod(self.image_shape) * 3  # image has 3 channels
        self.pickup_eps = -0.36
        self.task = task

        self._object_position_low = (.65, -0.1, -.36)
        self._object_position_high = (.75, 0.1, -.36)
        self._fixed_object_position = (.75, 0.2, -.36)
        self._success_threshold = success_threshold
        self.obs_img_dim = obs_img_dim #+.15
        self._view_matrix_obs = bullet.get_view_matrix(
            target_pos=[.7, 0, -0.3], distance=0.3,
            yaw=90, pitch=-15, roll=0, up_axis_index=2)
        self._projection_matrix_obs = bullet.get_projection_matrix(
            self.obs_img_dim, self.obs_img_dim)

        self.dt = 0.1
        super().__init__(*args, **kwargs)


    def _set_spaces(self):
        act_dim = 4
        act_bound = 1
        act_high = np.ones(act_dim) * act_bound
        self.action_space = gym.spaces.Box(-act_high, act_high)
        
        observation_dim = 11
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



    def _load_meshes(self):
        if self._invisible_robot:
            self._sawyer = bullet.objects.sawyer_invisible()
        else:
            self._sawyer = bullet.objects.sawyer_finger_visual_only()
        self._table = bullet.objects.table()
        self._objects = {}
        self._sensors = {}
        self._workspace = bullet.Sensor(self._sawyer,
            xyz_min=self._pos_low, xyz_max=self._pos_high,
            visualize=False, rgba=[0,1,0,.1])
        self._end_effector = bullet.get_index_by_attribute(
            self._sawyer, 'link_name', 'gripper_site')
        if self._randomize:
            object_position = np.random.uniform(
                low=self._object_position_low, high=self._object_position_high)
        else:
            object_position = self._fixed_object_position
        self._objects = {
            'lego': bullet.objects.lego(pos=object_position)
        }

        # Allow the objects to settle down after they are dropped in sim
        for _ in range(50):
            bullet.step()


    def step(self, *action):
        delta_pos, gripper = self._format_action(*action)
        pos = bullet.get_link_state(self._sawyer, self._end_effector, 'pos')
        pos += delta_pos * self._action_scale
        pos = np.clip(pos, self._pos_low, self._pos_high)

        self._simulate(pos, self.theta, gripper)
        if self._visualize: self.visualize_targets(pos)

        observation = self.get_observation()
        info = self.get_info()
        reward = self.get_reward(info)
        done = False
        self._prev_pos = bullet.get_link_state(self._sawyer, self._end_effector, 'pos')
        return observation, reward, done, info

    def get_info(self):
        object_pos = np.asarray(self.get_object_midpoint('lego'))
        height = object_pos[2]
        object_goal_distance = np.linalg.norm(object_pos - self.goal_pos)
        end_effector_pos = self.get_end_effector_pos()
        object_gripper_distance = np.linalg.norm(
            object_pos - end_effector_pos)
        gripper_goal_distance = np.linalg.norm(
            self.goal_pos - end_effector_pos)
        object_goal_success = int(object_goal_distance < self._success_threshold)
        picked_up = height > self.pickup_eps

        info = {
            # 'object_gripper_distance': object_gripper_distance,
            # 'gripper_goal_distance': gripper_goal_distance,
            'object_goal_distance': object_goal_distance,
            'object_goal_success': object_goal_success,
            'object_height': height,
            'picked_up': picked_up,
        }

        return info

    def get_diagnostics(self, paths):
        diagnostics = OrderedDict()
        state_key = "state_observation"
        goal_key = "state_desired_goal"
        values = []
        for i in range(len(paths)):
            state = paths[i]["observations"][-1][state_key][4:7]
            goal = paths[i]["observations"][-1][goal_key][4:7]
            distance = np.linalg.norm(state - goal)
            values.append(distance)
        diagnostics_key = goal_key + "/final/distance"
        diagnostics.update(create_stats_ordered_dict(
            diagnostics_key,
            values,
        ))

        values = []
        for i in range(len(paths)):
            for j in range(len(paths[i]["observations"])):
                state = paths[i]["observations"][j][state_key][4:7]
                goal = paths[i]["observations"][j][goal_key][4:7]
                distance = np.linalg.norm(state - goal)
                values.append(distance)
        diagnostics_key = goal_key + "/distance"
        diagnostics.update(create_stats_ordered_dict(
            diagnostics_key,
            values,
        ))
        return diagnostics

    def get_contextual_diagnostics(self, paths, contexts):
        diagnostics = OrderedDict()
        state_key = "state_observation"
        goal_key = "state_desired_goal"
        values = []
        for i in range(len(paths)):
            state = paths[i]["observations"][-1][state_key][4:7]
            goal = contexts[i][goal_key][4:7]
            distance = np.linalg.norm(state - goal)
            values.append(distance)
        diagnostics_key = goal_key + "/final/distance"
        diagnostics.update(create_stats_ordered_dict(
            diagnostics_key,
            values,
        ))

        values = []
        for i in range(len(paths)):
            for j in range(len(paths[i]["observations"])):
                state = paths[i]["observations"][j][state_key][4:7]
                goal = contexts[i][goal_key][4:7]
                distance = np.linalg.norm(state - goal)
                values.append(distance)
        diagnostics_key = goal_key + "/distance"
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
        self.goal_pos = goal['state_desired_goal'][4:7]

    def get_image(self, width, height):
        image = np.float32(self.render_obs())
        return image

    def get_reward(self, info):
        if self.task == 'goal_reaching':
            return info['object_goal_success'] - 1
        elif self.task == 'pickup':
            return info['picked_up'] - 1

    def reset(self):
        bullet.reset()
        bullet.setup_headless(self._timestep, solver_iterations=self._solver_iterations)
        self._load_meshes()
        self._format_state_query()

        self._prev_pos = np.array(self._pos_init)
        self.goal_pos = np.random.uniform(low=self._pos_low, high=self._pos_high)

        bullet.position_control(self._sawyer, self._end_effector, self._prev_pos, self.theta)
        # self._reset_hook(self)
        for _ in range(3):
            self.step([0.,0.,0.,-1])
        return self.get_observation()

    def compute_reward_pu(self, obs, actions, next_obs, contexts):
        height = self.format_obs(next_obs['state_observation'])[:, 4:7][:, 2]
        reward = (height > self.pickup_eps) - 1
        return reward
    
    def compute_reward_gr(self, obs, actions, next_obs, contexts):
        obj_state = self.format_obs(next_obs['state_observation'])[:, 4:7]
        obj_goal = self.format_obs(contexts['state_desired_goal'])[:, 4:7]
        object_goal_distance = np.linalg.norm(obj_state - obj_goal, axis=1)
        object_goal_success = object_goal_distance < self._success_threshold
        return object_goal_success - 1

    def compute_reward(self, obs, actions, next_obs, contexts):
        if self.task == 'goal_reaching':
            return self.compute_reward_gr(obs, actions, next_obs, contexts)
        elif self.task == 'pickup':
            return self.compute_reward_pu(obs, actions, next_obs, contexts)

    def get_observation(self):
        left_tip_pos = bullet.get_link_state(
            self._sawyer, 'right_gripper_l_finger_joint', keys='pos')
        right_tip_pos = bullet.get_link_state(
            self._sawyer, 'right_gripper_r_finger_joint', keys='pos')
        left_tip_pos = np.asarray(left_tip_pos)
        right_tip_pos = np.asarray(right_tip_pos)

        gripper_tips_distance = [np.linalg.norm(
            left_tip_pos - right_tip_pos)]
        end_effector_pos = self.get_end_effector_pos()

        object_info = bullet.get_body_info(self._objects['lego'],
                                           quat_to_deg=False)
        object_pos = object_info['pos']
        object_theta = object_info['theta']
        
        observation = np.concatenate(
            (end_effector_pos, gripper_tips_distance,
             object_pos, object_theta))

        goal_pos = np.concatenate(
            (self.goal_pos, gripper_tips_distance,
             self.goal_pos, object_theta))

        obs_dict = dict(
            observation=observation,
            state_observation=observation,
            desired_goal=goal_pos,
            state_desired_goal=goal_pos,
            achieved_goal=observation,
            state_achieved_goal=observation,
            )

        return obs_dict