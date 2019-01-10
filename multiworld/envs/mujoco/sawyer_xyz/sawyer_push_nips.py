from collections import OrderedDict
import numpy as np
from gym.spaces import Box, Dict
import mujoco_py

from multiworld.core.serializable import Serializable
from multiworld.envs.env_util import (
    get_stat_in_paths,
    create_stats_ordered_dict,
    get_asset_full_path,
)

from multiworld.envs.mujoco.mujoco_env import MujocoEnv
import copy

from multiworld.core.multitask_env import MultitaskEnv


class SawyerPushAndReachXYEnv(MujocoEnv, Serializable, MultitaskEnv):
    def __init__(
            self,
            frame_skip=50,
            action_scale=2./100,

            hand_low=(-0.2, 0.5),
            hand_high=(0.2, 0.7),

            puck_low=(-0.2, 0.5),
            puck_high=(0.2, 0.7),

            fix_goal=False,
            fixed_goal=(0.0, 0.55, 0.0, 0.65),
            goal_low=None,
            goal_high=None,

            fix_reset=False,
            fixed_hand_xyz=(0, 0.5, 0.06),
            reset_low=None,
            reset_high=None,

            hand_z_position=0.06,
            puck_z_position=0.02,
            puck_radius=.04,
            ee_radius=.015,

            reward_type='state_distance',
            norm_order=2,
            indicator_threshold=0.06,

            **kwargs
    ):
        self.quick_init(locals())
        MujocoEnv.__init__(self, self.model_name, frame_skip=frame_skip)

        hand_low = np.array(hand_low)
        hand_high = np.array(hand_high)
        mocap_low = hand_low
        mocap_high = hand_high
        self.mocap_low = np.hstack((mocap_low, np.array([0.0])))
        self.mocap_high = np.hstack((mocap_high, np.array([0.5])))
        puck_low = np.array(puck_low)
        puck_high = np.array(puck_high)
        self.obs_space = Box(
            np.hstack((hand_low, puck_low)),
            np.hstack((hand_high, puck_high)),
            dtype=np.float32
        )
        self.hand_space = Box(hand_low, hand_high, dtype=np.float32)
        self.puck_space = Box(puck_low, puck_high, dtype=np.float32)
        if goal_low is None:
            goal_low = self.obs_space.low.copy()
        if goal_high is None:
            goal_high = self.obs_space.high.copy()
        if reset_low is None:
            reset_low = self.obs_space.low.copy()
        if reset_high is None:
            reset_high = self.obs_space.high.copy()
        goal_low = np.array(goal_low)
        goal_high = np.array(goal_high)
        reset_low = np.array(reset_low)
        reset_high = np.array(reset_high)
        self.goal_space = Box(goal_low, goal_high, dtype=np.float32)
        self.reset_space = Box(reset_low, reset_high, dtype=np.float32)
        self.observation_space = Dict([
            ('observation', self.obs_space),
            ('desired_goal', self.goal_space),
            ('achieved_goal', self.obs_space),
            ('state_observation', self.obs_space),
            ('state_desired_goal', self.goal_space),
            ('state_achieved_goal', self.obs_space),
            ('proprio_observation', self.hand_space),
            ('proprio_desired_goal', Box(goal_low[:2], goal_high[:2], dtype=np.float32)),
            ('proprio_achieved_goal', self.hand_space),
        ])

        self.fix_reset = fix_reset
        self.fix_goal = fix_goal
        self.fixed_goal = np.array(fixed_goal)
        self._state_goal = None

        self.reward_type = reward_type
        self.norm_order = norm_order
        self.indicator_threshold = indicator_threshold

        self.action_space = Box(np.array([-1, -1]), np.array([1, 1]), dtype=np.float32)
        self._action_scale = action_scale
        self.hand_z_position = hand_z_position
        self.puck_z_position = puck_z_position
        self.fixed_hand_xyz = np.array(fixed_hand_xyz)
        self.reset_counter = 0
        self.puck_radius=puck_radius
        self.ee_radius = ee_radius
        self.reset()

    @property
    def model_name(self):
        return get_asset_full_path(
            'sawyer_xyz/sawyer_push_and_reach_mocap_goal_hidden.xml'
        )

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.distance = 1.0

        # robot view
        # rotation_angle = 90
        # cam_dist = 1
        # cam_pos = np.array([0, 0.5, 0.2, cam_dist, -45, rotation_angle])

        # 3rd person view
        cam_dist = 0.3
        rotation_angle = 270
        cam_pos = np.array([0, 1.0, 0.5, cam_dist, -45, rotation_angle])

        # top down view
        # cam_dist = 0.2
        # rotation_angle = 0
        # cam_pos = np.array([0, 0, 1.5, cam_dist, -90, rotation_angle])

        for i in range(3):
            self.viewer.cam.lookat[i] = cam_pos[i]
        self.viewer.cam.distance = cam_pos[3]
        self.viewer.cam.elevation = cam_pos[4]
        self.viewer.cam.azimuth = cam_pos[5]
        self.viewer.cam.trackbodyid = -1

    def step(self, action):
        action = np.clip(action, -1, 1)
        mocap_delta_z = self.hand_z_position - self.data.mocap_pos[0, 2]
        new_mocap_action = np.hstack((
            action,
            np.array([mocap_delta_z])
        ))
        self.mocap_set_action(new_mocap_action[:3] * self._action_scale)
        u = np.zeros(7)
        self.do_simulation(u, self.frame_skip)
        ob = self._get_obs()
        reward = self.compute_reward(action, ob)
        info = self._get_info()
        done = False
        return ob, reward, done, info

    def mocap_set_action(self, action):
        pos_delta = action[None]
        new_mocap_pos = self.data.mocap_pos + pos_delta
        new_mocap_pos[0, :] = np.clip(
            new_mocap_pos[0, :],
            self.mocap_low,
            self.mocap_high
        )
        self.data.set_mocap_pos('mocap', new_mocap_pos)
        self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))

    def _get_info(self):
        hand_goal = self._state_goal[:2]
        puck_goal = self._state_goal[-2:]
        hand_pos = self.get_endeff_pos()[:2]
        puck_pos = self.get_puck_pos()[:2]

        # hand distance
        hand_diff = hand_goal - hand_pos
        hand_distance = np.linalg.norm(hand_diff, ord=self.norm_order)
        hand_distance_l2 = np.linalg.norm(hand_diff, 2)

        # puck distance
        puck_diff = puck_goal - puck_pos
        puck_distance = np.linalg.norm(puck_diff, ord=self.norm_order)
        puck_distance_l2 = np.linalg.norm(puck_diff, 2)

        # touch distance
        touch_diff = self.get_endeff_pos() - self.get_puck_pos()
        touch_distance = np.linalg.norm(touch_diff, ord=self.norm_order)
        touch_distance_l2 = np.linalg.norm(touch_diff, ord=2)


        # state distance
        state_diff = np.hstack((hand_pos, puck_pos)) - self._state_goal
        state_distance = np.linalg.norm(state_diff, ord=self.norm_order)
        state_distance_l2 = np.linalg.norm(state_diff, ord=2)

        return dict(
            hand_distance=hand_distance, hand_distance_l2=hand_distance_l2,
            puck_distance=puck_distance, puck_distance_l2=puck_distance_l2,
            touch_distance=touch_distance, touch_distance_l2=touch_distance_l2,
            state_distance=state_distance, state_distance_l2=state_distance_l2,
            hand_success=float(hand_distance < self.indicator_threshold),
            puck_success=float(puck_distance < self.indicator_threshold),
            hand_and_puck_success=float(
                hand_distance+puck_distance < self.indicator_threshold
            ),
            touch_success=float(touch_distance < self.indicator_threshold),
            state_success=float(state_distance < self.indicator_threshold),
        )

    def _get_obs(self):
        e = self.get_endeff_pos()[:2]
        b = self.get_puck_pos()[:2]
        flat_obs = np.concatenate((e, b))

        return dict(
            observation=flat_obs,
            desired_goal=self._state_goal,
            achieved_goal=flat_obs,
            state_observation=flat_obs,
            state_desired_goal=self._state_goal,
            state_achieved_goal=flat_obs,
            proprio_observation=flat_obs[:2],
            proprio_desired_goal=self._state_goal[:2],
            proprio_achieved_goal=flat_obs[:2],
        )

    def compute_rewards(self, actions, obs):
        achieved_goals = obs['state_achieved_goal']
        desired_goals = obs['state_desired_goal']
        hand_pos = achieved_goals[:, :2]
        puck_pos = achieved_goals[:, -2:]
        hand_goals = desired_goals[:, :2]
        puck_goals = desired_goals[:, -2:]

        hand_distances = np.linalg.norm(hand_goals - hand_pos, ord=self.norm_order, axis=1)
        puck_distances = np.linalg.norm(puck_goals - puck_pos, ord=self.norm_order, axis=1)
        touch_distances = np.linalg.norm(hand_pos - puck_pos, ord=self.norm_order, axis=1)

        if self.reward_type == 'hand_distance':
            r = -hand_distances
        elif self.reward_type == 'hand_success':
            r = -(hand_distances > self.indicator_threshold).astype(float)
        elif self.reward_type == 'puck_distance':
            r = -puck_distances
        elif self.reward_type == 'puck_success':
            r = -(puck_distances > self.indicator_threshold).astype(float)
        elif self.reward_type == 'state_distance':
            r = -np.linalg.norm(
                achieved_goals - desired_goals,
                ord=self.norm_order,
                axis=1
            )
        elif self.reward_type == 'vectorized_state_distance':
            r = -np.abs(achieved_goals - desired_goals)
        elif self.reward_type == 'touch_distance':
            r = -touch_distances
        elif self.reward_type == 'touch_success':
            r = -(touch_distances > self.indicator_threshold).astype(float)
        else:
            raise NotImplementedError("Invalid/no reward type.")
        return r

    def get_diagnostics(self, paths, prefix=''):
        statistics = OrderedDict()
        for stat_name in [
            'hand_distance', 'hand_distance_l2',
            'puck_distance', 'puck_distance_l2',
            'state_distance', 'state_distance_l2',
            'touch_distance', 'touch_distance_l2',
            'hand_success',
            'puck_success',
            'hand_and_puck_success',
            'state_success',
            'touch_success',
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

    def get_puck_pos(self):
        return self.data.body_xpos[self.puck_id].copy()

    def get_endeff_pos(self):
        return self.data.body_xpos[self.endeff_id].copy()

    @property
    def endeff_id(self):
        return self.model.body_names.index('leftclaw')

    @property
    def puck_id(self):
        return self.model.body_names.index('puck')

    def reset(self):
        ob = self.reset_model()
        if self.viewer is not None:
            self.viewer_setup()
        return ob

    def reset_model(self):
        self._reset_hand()
        self._reset_puck()

        goal = self.sample_valid_goal()
        self.set_goal(goal)
        self.reset_counter += 1
        self.reset_mocap_welds()
        return self._get_obs()

    def _reset_hand(self):
        velocities = self.data.qvel.copy()
        angles = np.array(self.init_angles)
        self.set_state(angles.flatten(), velocities.flatten())

        if self.fix_reset:
            new_mocap_pos = self.fixed_hand_xyz.copy()
        else:
            new_mocap_pos_xy = np.random.uniform(self.reset_space.low[:2], self.reset_space.high[:2])
            new_mocap_pos = np.hstack((new_mocap_pos_xy, np.array([self.hand_z_position]))) #0.02
        for _ in range(10):
            self.data.set_mocap_pos('mocap', new_mocap_pos)
            self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
            self.do_simulation(None, self.frame_skip)

    def _reset_puck(self):
        self._set_puck_xy(self.sample_puck_xy())

        while not (self.puck_space.contains(self.get_puck_pos()[:2])) \
                or (self.end_effector_puck_collision(self.get_endeff_pos()[:2], self.get_puck_pos()[:2])):
            self._set_puck_xy(self.sample_puck_xy())

    def sample_puck_xy(self):
        if self.fix_reset:
            return np.array([0, 0.6])
        else:
            return np.random.uniform(self.reset_space.low[-2:], self.reset_space.high[-2:])

    def end_effector_puck_collision(self, ee, puck):
        dist = np.linalg.norm(ee - puck)
        return dist <= (self.puck_radius + self.ee_radius)

    def get_goal(self):
        return {
            'desired_goal': self._state_goal,
            'state_desired_goal': self._state_goal,
        }

    def set_goal(self, goal):
        self._state_goal = goal['state_desired_goal']
        hand_goal = self._state_goal[:2]
        puck_goal = self._state_goal[-2:]
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[14:17] = np.hstack((hand_goal.copy(), np.array([self.hand_z_position]))) #0.02
        qvel[14:17] = [0, 0, 0]
        qpos[21:24] = np.hstack((puck_goal.copy(), np.array([self.puck_z_position]))) #0.02
        qvel[21:24] = [0, 0, 0]
        self.set_state(qpos, qvel)

    def sample_valid_goal(self):
        goal = self.sample_goal()
        hand_goal_xy = goal['state_desired_goal'][:2]
        puck_goal_xy = goal['state_desired_goal'][-2:]
        dist = np.linalg.norm(hand_goal_xy - puck_goal_xy)
        while (dist <= self.puck_radius + self.ee_radius):
            goal = self.sample_goal()
            hand_goal_xy = goal['state_desired_goal'][:2]
            puck_goal_xy = goal['state_desired_goal'][-2:]
            dist = np.linalg.norm(hand_goal_xy - puck_goal_xy)
        return goal

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

    def set_to_goal(self, goal):
        state_goal = goal['state_desired_goal']
        self._set_hand_xy(state_goal[:2])
        self._set_puck_xy(state_goal[-2:])

    def _set_hand_xy(self, xy):
        for _ in range(10):
            self.data.set_mocap_pos('mocap', np.array([xy[0], xy[1], self.hand_z_position])) #0.02
            self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
            u = np.zeros(7)
            self.do_simulation(u, self.frame_skip)

    def _set_puck_xy(self, pos):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[7:10] = np.hstack((pos.copy(), np.array([self.puck_z_position]))) #0.02
        qvel[7:10] = [0, 0, 0]
        self.set_state(qpos, qvel)

    def get_env_state(self):
        joint_state = self.sim.get_state()
        mocap_state = self.data.mocap_pos, self.data.mocap_quat
        base_state = joint_state, mocap_state
        base_state = copy.deepcopy(base_state)
        goal = self._state_goal.copy()
        return base_state, goal

    def set_env_state(self, state):
        base_state, goal = state
        joint_state, mocap_state = base_state
        self.sim.set_state(joint_state)
        mocap_pos, mocap_quat = mocap_state
        self.data.set_mocap_pos('mocap', mocap_pos)
        self.data.set_mocap_quat('mocap', mocap_quat)
        self.set_goal({'state_desired_goal': goal})

    def reset_mocap_welds(self):
        """Resets the mocap welds that we use for actuation."""
        sim = self.sim
        if sim.model.nmocap > 0 and sim.model.eq_data is not None:
            for i in range(sim.model.eq_data.shape[0]):
                if sim.model.eq_type[i] == mujoco_py.const.EQ_WELD:
                    sim.model.eq_data[i, :] = np.array(
                        [0., 0., 0., 1., 0., 0., 0.])
        sim.forward()

    def reset_mocap2body_xpos(self):
        # move mocap to weld joint
        self.data.set_mocap_pos(
            'mocap',
            np.array([self.data.body_xpos[self.endeff_id]]),
        )
        self.data.set_mocap_quat(
            'mocap',
            np.array([self.data.body_xquat[self.endeff_id]]),
        )

    @property
    def init_angles(self):
        return [1.78026069e+00, - 6.84415781e-01, - 1.54549231e-01,
                2.30672090e+00, 1.93111471e+00, 1.27854012e-01,
                1.49353907e+00, 1.80196716e-03, 7.40415706e-01,
                2.09895360e-02, 9.99999990e-01, 3.05766105e-05,
                - 3.78462492e-06, 1.38684523e-04, - 3.62518873e-02,
                6.13435141e-01, 2.09686080e-02, 7.07106781e-01,
                1.48979724e-14, 7.07106781e-01, - 1.48999170e-14,
                0, 0.6, 0.02,
                1, 0, 1, 0,
                ]


class SawyerPushAndReachXYEasyEnv(SawyerPushAndReachXYEnv):
    """
    Always start the block in the same position, and use a 40x20 puck space
    """

    def __init__(
            self,
            **kwargs
    ):
        self.quick_init(locals())
        SawyerPushAndReachXYEnv.__init__(
            self,
            puck_goal_low=(-0.2, 0.5),
            puck_goal_high=(0.2, 0.7),
            **kwargs
        )

    def sample_puck_xy(self):
        return np.array([0, 0.6])


class SawyerPushAndReachXYHarderEnv(SawyerPushAndReachXYEnv):
    """
    Fixed initial position, all spaces are 40cm x 20cm
    """

    def __init__(
            self,
            **kwargs
    ):
        self.quick_init(locals())
        SawyerPushAndReachXYEnv.__init__(
            self,
            hand_goal_low=(-0.2, 0.5),
            hand_goal_high=(0.2, 0.7),
            puck_goal_low=(-0.2, 0.5),
            puck_goal_high=(0.2, 0.7),
            mocap_low=(-0.2, 0.5, 0.0),
            mocap_high=(0.2, 0.7, 0.5),
            **kwargs
        )

    def sample_puck_xy(self):
        return np.array([0, 0.6])
