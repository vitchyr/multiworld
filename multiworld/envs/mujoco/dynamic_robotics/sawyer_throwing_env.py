from collections import OrderedDict
import numpy as np
from multiworld.envs.mujoco.mujoco_env import MujocoEnv
from gym.spaces import Box, Dict
from multiworld.core.serializable import Serializable
from multiworld.core.multitask_env import MultitaskEnv
from multiworld.envs.env_util import get_stat_in_paths, \
    create_stats_ordered_dict, get_asset_full_path


class SawyerThrowingEnv(MujocoEnv, Serializable, MultitaskEnv):
    """Implements a torque-controlled Sawyer environment"""

    def __init__(self,
                 frame_skip=30,
                 action_scale=10,
                 keep_vel_in_obs=True,
                 use_safety_box=False,
                 fix_goal=False,
                 reward_type='obj_distance',
                 indicator_threshold=.05,
                 goal_low=None,
                 goal_high=None,
                 ):
        self.quick_init(locals())
        MultitaskEnv.__init__(self)
        self.action_scale = action_scale
        MujocoEnv.__init__(self, self.model_name, frame_skip=frame_skip)
        bounds = self.model.actuator_ctrlrange.copy()
        low = bounds[:7, 0]
        high = bounds[:7, 1]
        low = np.concatenate((low, [-1]))
        high = np.concatenate((high, [1]))
        self.action_space = Box(low=low, high=high)
        if goal_low is None:
            goal_low = np.array([1.2,  -0.25, 0.]) #in true position space
        else:
            goal_low = np.array(goal_low)

        if goal_high is None:
            goal_high = np.array([1.2,  0.25, 0.]) #in true position
        else:
            goal_high = np.array(goal_low)
        self.safety_box = Box(
            goal_low,
            goal_high
        )
        self.keep_vel_in_obs = keep_vel_in_obs
        self.goal_space = Box(goal_low, goal_high)
        obs_size = self._get_env_obs().shape[0]
        high = np.inf * np.ones(obs_size)
        low = -high
        self.obs_space = Box(low, high)
        self.achieved_goal_space = Box(
            -np.inf * np.ones(3),
            np.inf * np.ones(3)
        )
        self.observation_space = Dict([
            ('observation', self.obs_space),
            ('desired_goal', self.goal_space),
            ('achieved_goal', self.achieved_goal_space),
            ('state_observation', self.obs_space),
            ('state_desired_goal', self.goal_space),
            ('state_achieved_goal', self.achieved_goal_space),
        ])
        self.fix_goal = fix_goal
        self.goal_box_true_position = np.array([1.5, -.2, 0])
        self.fixed_goal = np.concatenate((np.array(self.init_angles[-2:]), [0]))+self.goal_box_true_position
        self.use_safety_box=use_safety_box
        self.prev_qpos = self.init_angles.copy()
        self.reward_type = reward_type
        self.indicator_threshold = indicator_threshold
        self.reset()

    @property
    def model_name(self):
       return get_asset_full_path('dynamic_robotics/sawyer_throwing.xml')

    def reset_to_prev_qpos(self):
        angles = self.data.qpos.copy()
        velocities = self.data.qvel.copy()
        angles[:] = self.prev_qpos.copy()
        velocities[:] = 0
        self.set_state(angles.flatten(), velocities.flatten())

    def is_outside_box(self):
        pos = self.get_endeff_pos()
        return not self.safety_box.contains(pos)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.distance = 1.0

        # 3rd person view
        cam_dist = 3
        rotation_angle = 180
        cam_pos = np.array([.5, 0, 0.3, cam_dist, -45, rotation_angle])

        for i in range(3):
            self.viewer.cam.lookat[i] = cam_pos[i]
        self.viewer.cam.distance = cam_pos[3]
        self.viewer.cam.elevation = cam_pos[4]
        self.viewer.cam.azimuth = cam_pos[5]
        self.viewer.cam.trackbodyid = -1

    def step(self, action):
        action[:7] = action[:7] * self.action_scale
        self.do_simulation(np.concatenate((action, [action[-1]*-1])), self.frame_skip)
        if self.use_safety_box:
            if self.is_outside_box():
                self.reset_to_prev_qpos()
            else:
                self.prev_qpos = self.data.qpos.copy()
        ob = self._get_obs()
        info = self._get_info()
        reward = self.compute_reward(action, ob)
        done = False
        return ob, reward, done, info

    def _get_env_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[:-2], #don't include the goal in the obs, assume higher level code appends goal when doing RL
            self.sim.data.qvel.flat,
            self.get_endeff_pos(),
        ])

    def _get_obs(self):
        state_obs = self._get_env_obs()

        obj_pos = state_obs[9:12]
        return dict(
            observation=state_obs,
            desired_goal=self._state_goal,
            achieved_goal=obj_pos,

            state_observation=state_obs,
            state_desired_goal=self._state_goal,
            state_achieved_goal=obj_pos,
        )

    def _get_info(self):
        obj_distance = np.linalg.norm(self._state_goal - self._get_env_obs()[9:12])
        return dict(
            obj_distance=obj_distance,
            obj_success=float(obj_distance < self.indicator_threshold),
        )

    def get_endeff_pos(self):
        return self.data.body_xpos[self.endeff_id].copy()

    def reset_model(self):
        angles = self.data.qpos.copy()
        velocities = self.data.qvel.copy()
        angles[:] = self.init_angles
        velocities[:] = 0
        goal_box_pos = self._state_goal - self.goal_box_true_position
        angles[-2:] = goal_box_pos[:2]
        self.set_state(angles.flatten(), velocities.flatten())
        self.sim.forward()
        self.prev_qpos=self.data.qpos.copy()

    def reset(self):
        self.set_goal(self.sample_goal())
        self.reset_model()
        self.sim.forward()
        self.prev_qpos = self.data.qpos.copy()
        return self._get_obs()

    @property
    def init_angles(self):
        return [
            -0.25, -.3+-6.95207647e-01, 0,
            .5+1.76670458e+00, 0, 0.3, 1.57,
            0, 0,
            0.44, 0.05, 0.04, 0., 0, 0, 0,
            -.3, .25
        ]

    @property
    def endeff_id(self):
        return self.model.body_names.index('right_hand')

    def get_diagnostics(self, paths, prefix=''):
        statistics = OrderedDict()
        for stat_name in [
            'obj_distance',
            'obj_success',
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

    """
    Multitask functions
    """
    @property
    def goal_dim(self) -> int:
        return 3

    def get_goal(self):
        return {
            'desired_goal': self._state_goal,
            'state_desired_goal': self._state_goal,
        }

    def get_goal_box_pos(self):
        return self.data.body_xpos[self.model.body_names.index('goal')].copy()

    def set_goal(self, goal):
        self._state_goal = goal['state_desired_goal']

    def sample_goals(self, batch_size):
        if self.fix_goal:
            goals = np.repeat(
                self.fixed_goal.copy()[None],
                batch_size,
                0
            )
        else:
            goals = np.random.uniform(
                self.goal_space.low,
                self.goal_space.high,
                size=(batch_size, self.goal_space.low.size),
            )
        return {
            'desired_goal': goals,
            'state_desired_goal': goals,
        }

    def compute_rewards(self, actions, obs):
        achieved_goals = obs['achieved_goal']
        desired_goals = obs['desired_goal']
        obj_pos = achieved_goals
        goals = desired_goals
        distances = np.linalg.norm(obj_pos - goals, axis=1)
        if self.reward_type == 'obj_distance':
            r = -distances
        elif self.reward_type == 'obj_success':
            r = -(distances > self.indicator_threshold).astype(float)
        else:
            raise NotImplementedError("Invalid/no reward type.")
        return r

    def get_env_state(self):
        joint_state = self.sim.get_state()
        goal = self._state_goal.copy()
        return joint_state, goal

    def set_env_state(self, state):
        state, goal = state
        self.sim.set_state(state)
        self.sim.forward()
        self._state_goal = goal

