import numpy as np
import gym
import pdb

import roboverse.bullet as bullet
from roboverse.envs.serializable import Serializable


class SawyerBaseEnv(gym.Env, Serializable):

    def __init__(self,
                 img_dim=256,
                 gui=False,
                 action_scale=.2,
                 action_repeat=10,
                 timestep=1./240, #1./240
                 solver_iterations=150,
                 gripper_bounds=[-1,1],
                 pos_init=[0.6, 0.0, -0.2], #[0.5, 0, 0]
                 pos_low=[0.6,-0.2,-.36], #[.4,-.6,-.36]
                 pos_high=[0.8,0.2,-0.2], #[1,.4,.25]
                 max_force=1000.,
                 visualize=True,
                 ):

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
        self._id = 'SawyerBaseEnv'

        self.theta = bullet.deg_to_quat([180, 0, 0])

        bullet.connect_headless(self._gui)
        # self.set_reset_hook()
        self._set_spaces()

        self._img_dim = img_dim
        self._view_matrix = bullet.get_view_matrix()
        self._projection_matrix = bullet.get_projection_matrix(self._img_dim, self._img_dim)

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

    # def get_constructor(self):
    #     return lambda: self.__class__(*self.args_, **self.kwargs_)

    def _set_spaces(self):
        act_dim = 4
        act_bound = 1
        act_high = np.ones(act_dim) * act_bound
        self.action_space = gym.spaces.Box(-act_high, act_high)

        obs = self.reset()
        observation_dim = len(obs)
        obs_bound = 100
        obs_high = np.ones(observation_dim) * obs_bound
        self.observation_space = gym.spaces.Box(-obs_high, obs_high)

    def reset(self):

        bullet.reset()
        bullet.setup_headless(self._timestep, solver_iterations=self._solver_iterations)
        self._load_meshes()
        self._format_state_query()


        self._prev_pos = np.array(self._pos_init)
        bullet.position_control(self._sawyer, self._end_effector, self._prev_pos, self.theta)
        # self._reset_hook(self)
        for _ in range(3):
            self.step([0.,0.,0.,-1])
        return self.get_observation()

    # def set_reset_hook(self, fn=lambda env: None):
    #     self._reset_hook = fn

    def open_gripper(self, act_repeat=10):
        delta_pos = [0,0,0]
        gripper = 0
        for _ in range(act_repeat):
            self.step(delta_pos, gripper)

    def get_body(self, name):
        if name == 'sawyer':
            return self._sawyer
        else:
            return self._objects[name]

    def get_object_midpoint(self, object_key):
        return bullet.get_midpoint(self._objects[object_key])

    def get_end_effector_pos(self):
        return bullet.get_link_state(self._sawyer, self._end_effector, 'pos')

    def _load_meshes(self):
        self._sawyer = bullet.objects.sawyer()
        self._table = bullet.objects.table()
        self._objects = {}
        self._sensors = {}
        self._workspace = bullet.Sensor(self._sawyer,
            xyz_min=self._pos_low, xyz_max=self._pos_high,
            visualize=False, rgba=[0,1,0,.1])
        self._end_effector = bullet.get_index_by_attribute(
            self._sawyer, 'link_name', 'gripper_site')


    def _format_state_query(self):
        ## position and orientation of body root
        bodies = [v for k,v in self._objects.items() if not bullet.has_fixed_root(v)]
        ## position and orientation of link
        links = [(self._sawyer, self._end_effector)]
        ## position and velocity of prismatic joint
        joints = [(self._sawyer, None)]
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
        pos = bullet.get_link_state(self._sawyer, self._end_effector, 'pos')
        pos += delta_pos * self._action_scale
        pos = np.clip(pos, self._pos_low, self._pos_high)

        self._simulate(pos, self.theta, gripper)
        if self._visualize: self.visualize_targets(pos)

        observation = self.get_observation()
        reward = self.get_reward(observation)
        done = self.get_termination(observation)
        self._prev_pos = bullet.get_link_state(self._sawyer, self._end_effector, 'pos')
        return observation, reward, done, {}

    def _simulate(self, pos, theta, gripper):
        for _ in range(self._action_repeat):
            bullet.sawyer_position_ik(
                self._sawyer, self._end_effector, 
                pos, theta,
                gripper, gripper_bounds=self._gripper_bounds, 
                discrete_gripper=False, max_force=self._max_force
            )
            bullet.step_ik()

    def render(self, mode='rgb_array'):
        img, depth, segmentation = bullet.render(
            self._img_dim, self._img_dim, self._view_matrix, self._projection_matrix)
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


