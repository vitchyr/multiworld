import numpy as np
import gym
import pdb

import roboverse.bullet as bullet
from roboverse.envs.serializable import Serializable


class WidowBaseEnv(gym.Env, Serializable):
    def __init__(self,
                 img_dim=48,
                 gui=False,
                 action_scale=.2,
                 action_repeat=10,
                 timestep=1. / 120,
                 solver_iterations=150,
                 gripper_bounds=[-1, 1],
                 pos_init=[0.5, 0, 0],
                 pos_high=[1, .4, .25],
                 pos_low=[.4, -.6, -.36],
                 max_force=1000.,
                 visualize=True,
                 downwards=False,
                 ):

        self._id = 'WidowBaseEnv'
        self._robot_name = 'widowx'
        self._gripper_joint_name = (
        'gripper_prismatic_joint_1', 'gripper_prismatic_joint_2')
        if self._env_name == 'WidowX200GraspEnv':
            self._gripper_joint_name = ('left_finger', 'right_finger')

        self._gripper_range = range(7, 9)
        self.downwards = downwards

        self._end_effector_link_name = 'gripper_rail_link'

        if self._env_name == 'WidowX200GraspEnv':
            self._end_effector_link_name = 'wx200/gripper_bar_link'

        self.obs_img_dim = img_dim
        self.image_shape = (img_dim, img_dim)
        self.image_length = img_dim*img_dim*3

        self._gui = gui
        self._action_scale = action_scale
        self._action_repeat = action_repeat
        self._timestep = timestep
        self._solver_iterations = solver_iterations
        self._gripper_bounds = gripper_bounds
        self._pos_init = pos_init
        self._pos_low = pos_low
        self._pos_high = pos_high
        self._max_force = max_force
        self._visualize = visualize
        self._img_dim = img_dim

        bullet.connect_headless(self._gui)
        self._load_meshes()

        view_matrix_args = dict(target_pos=[.95, -0.05, -0.2], distance=0.10,
                                yaw=90, pitch=-40, roll=0, up_axis_index=2)
        self._view_matrix = bullet.get_view_matrix(
            **view_matrix_args)
        self._projection_matrix = bullet.get_projection_matrix(
            self._img_dim, self._img_dim)

        self._view_matrix_obs = bullet.get_view_matrix(
            **view_matrix_args)
        self._projection_matrix_obs = bullet.get_projection_matrix(
            self._img_dim, self._img_dim)

        # self._setup_environment()

        self._end_effector = self._end_effector = bullet.get_index_by_attribute(
            self._robot_id, 'link_name', self._end_effector_link_name)
        self._set_spaces()


    def _load_meshes(self):
        if self.downwards:
            self._robot_id = bullet.objects.widow_downwards()
        else:
            # print("self._env_name", self._env_name)
            if self._env_name in ['WidowX200GraspEnv', 'Widow200GraspV2Env']:
                self._robot_id = bullet.objects.widowx_200()
            else:
                self._robot_id = bullet.objects.widow()
        self._table = bullet.objects.table()
        self._objects = {}
        self._workspace = bullet.Sensor(self._robot_id,
                                        xyz_min=self._pos_low,
                                        xyz_max=self._pos_high,
                                        visualize=False, rgba=[0, 1, 0, .1])

    def get_params(self):
        labels = ['_action_scale', '_action_repeat',
                  '_timestep', '_solver_iterations',
                  '_gripper_bounds', '_pos_low', '_pos_high', '_id']
        params = {label: getattr(self, label) for label in labels}
        return params

    @property
    def parallel(self):
        return False

    def check_params(self, other):
        params = self.get_params()
        assert set(params.keys()) == set(other.keys())
        for key, val in params.items():
            if val != other[key]:
                message = 'Found mismatch in {} | env : {} | demos : {}'.format(
                    key, val, other[key]
                )
                raise RuntimeError(message)

    def _set_spaces(self):
        act_dim = 4
        act_bound = 1
        act_high = np.ones(act_dim) * act_bound
        self.action_space = gym.spaces.Box(-act_high, act_high)

        self._format_state_query()
        obs = self.get_observation()
        observation_dim = len(obs)
        obs_bound = 100
        obs_high = np.ones(observation_dim) * obs_bound
        self.observation_space = gym.spaces.Box(-obs_high, obs_high)

    def reset(self):

        bullet.reset()
        self._load_meshes()
        # Allow the objects to settle down after they are dropped in sim
        for _ in range(50):
            bullet.step()

        self._format_state_query()

        bullet.setup_headless(self._timestep,
                              solver_iterations=self._solver_iterations)

        self._prev_pos = np.array(self._pos_init)
        # self.theta = bullet.deg_to_quat([180, 0, 0])
        self.theta = bullet.deg_to_quat([110, 85, 37])

        bullet.position_control(self._robot_id, self._end_effector,
                                self._prev_pos, self.theta)
        self.open_gripper()
        #self._reset_hook(self)
        return self.get_observation()

    def open_gripper(self, act_repeat=10):
        delta_pos = [0, 0, 0]
        gripper = 0
        for _ in range(act_repeat):
            self.step(delta_pos, gripper)

    def get_body(self, name):
        if name == self._robot_name:
            return self._robot_id
        else:
            return self._objects[name]

    def get_object_midpoint(self, object_key):
        return bullet.get_midpoint(self._objects[object_key])

    def get_end_effector_pos(self):
        return bullet.get_link_state(self._robot_id, self._end_effector, 'pos')

    def _format_state_query(self):
        ## position and orientation of body root
        bodies = [v for k, v in self._objects.items()
                  if not bullet.has_fixed_root(v)]
        ## position and orientation of link
        links = [(self._robot_id, self._end_effector)]
        ## position and velocity of prismatic joint
        joints = [(self._robot_id, None)]
        self._state_query = bullet.format_sim_query(bodies, links, joints)

    def _format_action(self, *action):
        if len(action) == 1:
            delta_pos, gripper = action[0][:-1], action[0][-1]
        elif len(action) == 2:
            delta_pos, gripper = action[0], action[1]
        else:
            raise RuntimeError('Unrecognized action: {}'.format(action))
        return np.array(delta_pos), gripper

    def get_observation(self):
        observation = bullet.get_sim_state(*self._state_query)
        return observation

    def step(self, *action):
        delta_pos, gripper = self._format_action(*action)
        pos = bullet.get_link_state(self._robot_id, self._end_effector, 'pos')
        pos += delta_pos * self._action_scale
        pos = np.clip(pos, self._pos_low, self._pos_high)

        self._simulate(pos, self.theta, gripper)
        if self._visualize: self.visualize_targets(pos)

        observation = self.get_observation()
        reward = self.get_reward(observation)
        done = self.get_termination(observation)
        self._prev_pos = bullet.get_link_state(self._robot_id,
                                               self._end_effector, 'pos')
        return observation, reward, done, {}

    def _simulate(self, pos, theta, gripper, discrete_gripper=True):
        for _ in range(self._action_repeat):
            bullet.sawyer_position_ik(
                self._robot_id, self._end_effector,
                pos, self.theta,
                gripper, gripper_name=self._gripper_joint_name,
                gripper_bounds=self._gripper_bounds,
                discrete_gripper=discrete_gripper, max_force=self._max_force
            )
            bullet.step_ik(self._gripper_range)

    def render(self, mode='rgb_array'):
        img, depth, segmentation = bullet.render(
            self._img_dim, self._img_dim, self._view_matrix,
            self._projection_matrix)
        return img

    def get_termination(self, observation):
        return False

    def get_reward(self, observation):
        return 0

    def visualize_targets(self, pos):
        bullet.add_debug_line(self._prev_pos, pos)

    def save_state(self, *save_path):
        state_id = bullet.save_state(*save_path)
        return state_id

    def load_state(self, load_path):
        bullet.load_state(load_path)
        obs = self.get_observation()
        return obs

    '''
        prevents always needing a gym adapter in softlearning
        @TODO : remove need for this method
    '''

    def convert_to_active_observation(self, obs):
        return obs


