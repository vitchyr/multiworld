from collections import OrderedDict
import numpy as np
import time
from gym.spaces import Box, Dict

from multiworld.envs.env_util import get_stat_in_paths, \
    create_stats_ordered_dict, get_asset_full_path
from multiworld.core.multitask_env import MultitaskEnv
from multiworld.envs.mujoco.sawyer_xyz.base import SawyerXYZEnv
from multiworld.envs.mujoco.cameras import sawyer_pick_and_place_camera

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from mujoco_py.builder import MujocoException

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


class SawyerPickAndPlaceEnv(MultitaskEnv, SawyerXYZEnv):
    def __init__(
            self,
            obj_low=None,
            obj_high=None,

            reward_type='hand_and_obj_distance',
            indicator_threshold=0.02,

            fix_goal=False,
            fixed_goal=None,
            goal_low=None,
            goal_high=None,
            hard_goals=False,

            hide_goal_markers=True,
            presampled_goals=None,
            num_goals_presampled=1000,
            norm_order=2,
            structure='2d',

            two_obj=False,
            frame_skip=100,
            reset_p=None,
            goal_p=None,

            fixed_reset=None,
            hide_state_markers=True,

            test_mode_case_num=None,
            **kwargs
    ):
        self.quick_init(locals())
        # self.obj_id[0] for first object, self.obj_id[1] for second object.
        self.obj_id = {
            0: '',
            1: '2',
        }

        assert structure in ['2d', '3d', '2d_wall_short', '2d_wall_tall', '2d_wall_tall_dark']
        self.structure = structure
        self.two_obj = two_obj
        self.hard_goals = hard_goals
        MultitaskEnv.__init__(self)
        SawyerXYZEnv.__init__(
            self,
            model_name=self.model_name,
            frame_skip=frame_skip,
            **kwargs
        )
        self.norm_order = norm_order
        if obj_low is None:
            obj_low = np.copy(self.hand_low)
            obj_low[2] = 0.0
        if obj_high is None:
            obj_high = np.copy(self.hand_high)
        self.obj_low = np.array(obj_low)
        self.obj_high = np.array(obj_high)
        if goal_low is None:
            goal_low = np.hstack((self.hand_low, obj_low))
        if goal_high is None:
            goal_high = np.hstack((self.hand_high, obj_high))

        self.reward_type = reward_type
        self.indicator_threshold = indicator_threshold

        self.subgoals = None
        self.fix_goal = fix_goal
        self._state_goal = None

        self.hide_goal_markers = hide_goal_markers
        self.hide_state_markers = hide_state_markers

        self.action_space = Box(
            np.array([-1, -1, -1, -1]),
            np.array([1, 1, 1, 1]),
            dtype=np.float32
        )
        if self.two_obj:
            self.hand_and_obj_space = Box(
                np.hstack((self.hand_low, obj_low, obj_low)),
                np.hstack((self.hand_high, obj_high, obj_high)),
                dtype=np.float32
            )
        else:
            self.hand_and_obj_space = Box(
                np.hstack((self.hand_low, obj_low)),
                np.hstack((self.hand_high, obj_high)),
                dtype=np.float32
            )
        self.hand_space = Box(
            self.hand_low,
            self.hand_high,
            dtype=np.float32
        )
        if self.two_obj:
            self.gripper_and_hand_and_obj_space = Box(
                np.hstack((self.hand_low, obj_low, obj_low, [0.0])),
                np.hstack((self.hand_high, obj_high, obj_high, [0.07])),
                dtype=np.float32
            )
        else:
            self.gripper_and_hand_and_obj_space = Box(
                np.hstack((self.hand_low, obj_low, [0.0])),
                np.hstack((self.hand_high, obj_high, [0.07])),
                dtype=np.float32
            )

        self.observation_space = Dict([
            ('observation', self.gripper_and_hand_and_obj_space),
            ('desired_goal', self.gripper_and_hand_and_obj_space),
            ('achieved_goal', self.gripper_and_hand_and_obj_space),
            ('state_observation', self.gripper_and_hand_and_obj_space),
            ('state_desired_goal', self.gripper_and_hand_and_obj_space),
            ('state_achieved_goal', self.gripper_and_hand_and_obj_space),
            ('proprio_observation', self.hand_space),
            ('proprio_desired_goal', self.hand_space),
            ('proprio_achieved_goal', self.hand_space),
        ])

        if presampled_goals is not None:
            self._presampled_goals = presampled_goals
            self.num_goals_presampled = len(list(self._presampled_goals.values)[0])
        else:
            # if num_goals_presampled > 0, presampled_goals will be created when sample_goal is first called
            self._presampled_goals = None
            self.num_goals_presampled = num_goals_presampled
        self.picked_up_object = False
        self.train_pickups = 0
        self.eval_pickups = 0
        self.cur_mode = 'train'

        self.obj_radius = np.array([0.018, 0.016]) #0.020
        self.ee_radius = np.array([0.053, 0.058]) #0.065
        if self.structure == '2d_wall_short':
            self.wall_radius = np.array([0.03, 0.02])
            self.wall_center = np.array([0.0, 0.60, 0.02])
        elif self.structure in ['2d_wall_tall', '2d_wall_tall_dark']:
            self.wall_radius = np.array([0.015, 0.04])
            self.wall_center = np.array([0.0, 0.60, 0.04])
        else:
            self.wall_radius = None
            self.wall_center = None
        # self.obj_radius = 0.020
        # self.ee_radius = 0.065
        if fixed_reset is not None:
            fixed_reset = np.array(fixed_reset)
        self.fixed_reset = fixed_reset
        if fixed_goal is not None:
            fixed_goal = np.array(fixed_goal)
        self.fixed_goal = fixed_goal
        self.reset_p = reset_p
        self.goal_p = goal_p

        self.test_mode_case_num = test_mode_case_num
        if self.test_mode_case_num == 1:
            self.fixed_reset = np.array([0.0, 0.50, 0.05, 0.0, 0.50, 0.015, 0.0, 0.70, 0.015])
            self.fixed_goal = np.array([0.0, 0.50, 0.10, 0.0, 0.70, 0.03, 0.0, 0.70, 0.015])
        elif self.test_mode_case_num == 2:
            self.fixed_reset = np.array([0.0, 0.50, 0.05, 0.0, 0.50, 0.015, 0.0, 0.70, 0.015])
            self.fixed_goal = np.array([0.0, 0.70, 0.10, 0.0, 0.50, 0.015, 0.0, 0.50, 0.03])
        elif self.test_mode_case_num == 3:
            self.fixed_reset = np.array([0.0, 0.60, 0.05, 0.0, 0.50, 0.015, 0.0, 0.70, 0.015])
            self.fixed_goal = np.array([0.0, 0.60, 0.05, 0.0, 0.70, 0.015, 0.0, 0.50, 0.015])

        if presampled_goals is not None:
            self.reset()
        else:
            self.num_goals_presampled = 1
            self.reset()
            self._presampled_goals = None
            self.num_goals_presampled = num_goals_presampled

    def set_xyz_action(self, action):
        action = np.clip(action, -1, 1)
        pos_delta = action * self.action_scale
        # print(self.get_endeff_pos() - self.data.mocap_pos)
        new_mocap_pos = self.get_endeff_pos() - pos_delta[None]
        # new_mocap_pos = self.data.mocap_pos - pos_delta[None]
        new_mocap_pos[0, :] = np.clip(
            new_mocap_pos[0, :],
            self.mocap_low,
            self.mocap_high,
        )
        self.data.set_mocap_pos('mocap', new_mocap_pos)
        self.data.set_mocap_quat('mocap', np.array([0, 0, 1, 0]))

    @property
    def model_name(self):
        if self.structure == '3d':
            return get_asset_full_path('sawyer_xyz/sawyer_pick_and_place_3d.xml')
        elif self.structure == '2d':
            return get_asset_full_path('sawyer_xyz/sawyer_pick_and_place_2d.xml')
        elif self.structure == '2d_wall_short':
            return get_asset_full_path('sawyer_xyz/sawyer_pick_and_place_2d_wall_short.xml')
        elif self.structure == '2d_wall_tall':
            return get_asset_full_path('sawyer_xyz/sawyer_pick_and_place_2d_wall_tall.xml')
        elif self.structure == '2d_wall_tall_dark':
            return get_asset_full_path('sawyer_xyz/sawyer_pick_and_place_2d_wall_tall_dark.xml')

    def train(self):
        self.cur_mode = 'train'

    def eval(self):
        self.cur_mode = 'eval'

    def mode(self, mode):
        if 'train' in mode:
            self.train()
        else:
            self.eval()

    def viewer_setup(self):
        sawyer_pick_and_place_camera(self.viewer.cam)

    def step(self, action):
        prev_ob = self._get_obs()
        self.set_xyz_action(action[:3])
        mujoco_exception_raised = False
        try:
            self.do_simulation(action[3:], self.frame_skip)
        except MujocoException as e:
            mujoco_exception_raised = True
            print("Inside step:", e)
        if (
                (self.get_obj_pos(obj_id=0)[2] > .07) or
                (self.two_obj and self.get_obj_pos(obj_id=1)[2] > .07)
        ):
            if not self.picked_up_object:
                if self.cur_mode == 'train':
                    self.train_pickups += 1
                else:
                    self.eval_pickups += 1
                self.picked_up_object = True
        # The marker seems to get reset every time you do a simulation
        self._set_goal_marker(self._state_goal)
        ob = self._get_obs()
        self._set_state_marker(ob["state_observation"])
        reward = self.compute_reward(action, ob, prev_obs=prev_ob)
        info = self._get_info(
            ob=ob,
            prev_ob=prev_ob,
            mujoco_exception_raised=mujoco_exception_raised
        )
        done = False
        return ob, reward, done, info

    def _get_obs(self):
        ee = self.get_endeff_pos()
        b0 = self.get_obj_pos(obj_id=0)
        if self.two_obj:
            b1 = self.get_obj_pos(obj_id=1)
        gripper = self.get_gripper_pos()
        if self.two_obj:
            flat_obs_with_gripper = np.concatenate((ee, b0, b1, gripper))
        else:
            flat_obs_with_gripper = np.concatenate((ee, b0, gripper))
        if self._state_goal is not None:
            hand_goal = self._state_goal[:3]
        else:
            hand_goal = None
        return dict(
            observation=flat_obs_with_gripper,
            desired_goal=self._state_goal,
            achieved_goal=flat_obs_with_gripper,
            state_observation=flat_obs_with_gripper,
            state_desired_goal=self._state_goal,
            state_achieved_goal=flat_obs_with_gripper,
            proprio_observation=ee,
            proprio_achieved_goal=ee,
            proprio_desired_goal=hand_goal,
        )

    def _get_info(self, ob, prev_ob, mujoco_exception_raised=False):
        hand_goal = self._state_goal[:3]
        obj_goal0 = self._state_goal[3:6]
        if self.two_obj:
            obj_goal1 = self._state_goal[6:9]
        obj_pos0 = self.get_obj_pos(obj_id=0)
        hand_distance = np.linalg.norm(hand_goal - self.get_endeff_pos(), ord=self.norm_order,)
        obj_distance0 = np.linalg.norm(obj_goal0 - obj_pos0, ord=self.norm_order,)
        if self.two_obj:
            obj_pos1 = self.get_obj_pos(obj_id=1)
            obj_distance1 = np.linalg.norm(obj_goal1 - obj_pos1, ord=self.norm_order,)
            obj_distance = obj_distance0 + obj_distance1
            hand_indicator_threshold = self.indicator_threshold * 2
        else:
            obj_distance = obj_distance0
            hand_indicator_threshold = self.indicator_threshold

        obj_diff0 = prev_ob['state_observation'][3:6] - ob['state_observation'][3:6]
        if self.two_obj:
            obj_diff1 = prev_ob['state_observation'][6:9] - ob['state_observation'][6:9]
        else:
            obj_diff1 = np.array([0, 0, 0])
        flying_object = False
        if np.linalg.norm(obj_diff0, ord=np.inf) > 0.20 or np.linalg.norm(obj_diff1, ord=np.inf) > 0.20:
            flying_object = True
            print("flying_object!")

        info = dict(
            hand_distance=hand_distance,
            obj_distance=obj_distance,
            obj_distance0=obj_distance0,
            obj_x0=np.abs(obj_pos0[0]),
            obj_y0=np.abs(obj_pos0[1]),
            obj_z0=np.abs(obj_pos0[2]),
            hand_and_obj_distance=hand_distance+obj_distance,
            hand_success=float(hand_distance < hand_indicator_threshold),
            obj_success=float(obj_distance < hand_indicator_threshold),
            hand_and_obj_success=float(
                hand_distance+obj_distance < self.indicator_threshold + hand_indicator_threshold
            ),
            total_pickups=self.train_pickups if self.cur_mode == 'train' else self.eval_pickups,
            mujoco_exception_raised=int(mujoco_exception_raised),
            flying_object=int(flying_object),
        )
        if self.two_obj:
            info["obj_distance1"] = obj_distance1
            info["obj_x1"] = np.abs(obj_pos1[0])
            info["obj_y1"] = np.abs(obj_pos1[1])
            info["obj_z1"] = np.abs(obj_pos1[2])
        return info

    def get_obj_pos(self, obj_id):
        return self.data.get_body_xpos('obj' + self.obj_id[obj_id]).copy()

    def _set_goal_marker(self, goal):
        """
        This should be use ONLY for visualization. Use self._state_goal for
        logging, learning, etc.
        """
        self.data.site_xpos[self.model.site_name2id('hand-goal-site')] = (
            goal[:3]
        )
        self.data.site_xpos[self.model.site_name2id('obj-goal-site')] = (
            goal[3:6]
        )
        if self.hide_goal_markers:
            self.data.site_xpos[self.model.site_name2id('hand-goal-site'), 2] = (
                -1000
            )
            self.data.site_xpos[self.model.site_name2id('obj-goal-site'), 2] = (
                -1000
            )

    def _set_state_marker(self, ob):
        if self.hide_state_markers:
            self.data.site_xpos[self.model.site_name2id('obj'), 2] = (
                -1000
            )
            if self.two_obj:
                self.data.site_xpos[self.model.site_name2id('obj2'), 2] = (
                    -1000
                )
        else:
            self.data.site_xpos[self.model.site_name2id('obj')] = (
                ob[3:6]
            )
            if self.two_obj:
                self.data.site_xpos[self.model.site_name2id('obj2')] = (
                    ob[6:9]
                )

    def _set_obj_xyz(self, pos, obj_id, set_vel_to_zero=True):
        if self.structure in ['3d']:
            qpos = self.data.qpos.flat.copy()
            qvel = self.data.qvel.flat.copy()
            obj_name = 'objjoint' + self.obj_id[obj_id]
            qpos_obj_start, qpos_obj_end = self.sim.model.get_joint_qpos_addr(obj_name)
            qvel_obj_start, qvel_obj_end = self.sim.model.get_joint_qvel_addr(obj_name)
            # the qpos is 7 dimensions. It might be x, y, z, quad. Ignore quad for now
            qpos[qpos_obj_start:qpos_obj_start + 3] = pos.copy()
            if set_vel_to_zero:
                qvel[qvel_obj_start:qvel_obj_end] = 0
            self.set_state(qpos, qvel)
        elif self.structure in ['2d', '2d_wall_short', '2d_wall_tall', '2d_wall_tall_dark']:
            qpos = self.data.qpos.flat.copy()
            qvel = self.data.qvel.flat.copy()
            obj_name = 'objjoint_y' + self.obj_id[obj_id]
            qpos_idx = self.sim.model.get_joint_qpos_addr(obj_name)
            qvel_idx = self.sim.model.get_joint_qvel_addr(obj_name)
            # the qpos is 7 dimensions. It might be x, y, z, quad. Ignore quad for now
            qpos[qpos_idx] = pos[1]
            if set_vel_to_zero:
                qvel[qvel_idx] = 0
            obj_name = 'objjoint_z' + self.obj_id[obj_id]
            qpos_idx = self.sim.model.get_joint_qpos_addr(obj_name)
            qvel_idx = self.sim.model.get_joint_qvel_addr(obj_name)
            # the qpos is 7 dimensions. It might be x, y, z, quad. Ignore quad for now
            qpos[qpos_idx] = pos[2]
            if set_vel_to_zero:
                qvel[qvel_idx] = 0
            self.set_state(qpos, qvel)

    def reset_model(self):
        type = self._sample_reset_type()

        self._reset_ee()
        self._reset_obj(type=type)
        self._reset_gripper(type=type)

        ob = self._get_obs()
        self._set_state_marker(ob["state_observation"])

        self.set_goal(self.sample_goal())

        self.picked_up_object = False
        return self._get_obs()

    def _sample_reset_type(self):
        if self.two_obj:
            if self.reset_p is not None:
                p = self.reset_p
            else:
                p = [1 / 3, 1 / 3, 1 / 3]
            return np.random.choice(['ground', 'stacked', 'air'], size=1, p=p)[0]
        else:
            if self.reset_p is not None:
                p = self.reset_p
            else:
                p = [1 / 2, 1 / 2]
            return np.random.choice(['ground', 'air'], size=1, p=p)[0]

    def _sample_goal_type(self):
        if self.two_obj:
            if self.goal_p is not None:
                p = self.goal_p
            else:
                p = [1 / 3, 1 / 3, 1 / 3]
            return np.random.choice(['ground', 'stacked', 'air'], size=1, p=p)[0]
        else:
            if self.goal_p is not None:
                p = self.goal_p
            else:
                p = [1 / 2, 1 / 2]
            return np.random.choice(['ground', 'air'], size=1, p=p)[0]

    def _sample_realistic_goals(self, batch_size):
        print("sampling goals from pick_n_place env:", batch_size)
        t = time.time()
        goals = np.zeros((batch_size, self.observation_space.spaces['state_desired_goal'].low.size))
        for i in range(batch_size):
            type = self._sample_goal_type()
            ee = self._sample_realistic_ee()
            obj = self._sample_realistic_obj(type=type, ee=ee)
            gripper = self._sample_realistic_gripper(type=type)
            goals[i] = np.concatenate((ee, obj, gripper))

        pre_state = self.get_env_state()
        for i in range(batch_size):
            self.set_to_goal({'state_desired_goal': goals[i]})
            goals[i] = self._get_obs()['state_achieved_goal']
        self.set_env_state(pre_state)
        print("total time:", time.time()-t)

        return {
            'desired_goal': goals,
            'state_desired_goal': goals,
            'proprio_desired_goal': goals[:, :3],
        }

    def set_to_goal(self, goal):
        ### WILL NOT WORK FOR AIR GOALS
        state_goal = goal['state_desired_goal']

        if 'wall' in self.structure:
            self.data.set_mocap_pos('mocap', np.array([0.0, 0.60, 0.30]))
            self.data.set_mocap_quat('mocap', np.array([0, 0, 1, 0]))
            try:
                self.do_simulation(np.array([-1]), self.frame_skip)
            except MujocoException as e:
                print("Inside set_to_goal:", e)

        for _ in range(1):  # 10
            self.data.set_mocap_pos('mocap', state_goal[:3])
            self.data.set_mocap_quat('mocap', np.array([0, 0, 1, 0]))
            try:
                self.do_simulation(np.array([-1]), self.frame_skip)
            except MujocoException as e:
                print("Inside set_to_goal:", e)
        self._set_obj_xyz(state_goal[3:6], obj_id=0)
        if self.two_obj:
            self._set_obj_xyz(state_goal[6:9], obj_id=1)

        for _ in range(4):
            try:
                self.do_simulation(state_goal[-1:], self.frame_skip)
            except MujocoException as e:
                print("Inside set_to_goal:", e)

        ob = self._get_obs()
        self._set_state_marker(ob["state_observation"])

    def _reset_ee(self):
        new_mocap_pos_xy = self._sample_realistic_ee(mode='reset')

        if 'wall' in self.structure:
            self.data.set_mocap_pos('mocap', [0.0, 0.60, 0.30])
            self.data.set_mocap_quat('mocap', np.array([0, 0, 1, 0]))
            try:
                self.do_simulation(None, self.frame_skip)
            except MujocoException as e:
                print("Inside _reset_ee:", e)

        for _ in range(1): #10
            self.data.set_mocap_pos('mocap', new_mocap_pos_xy)
            self.data.set_mocap_quat('mocap', np.array([0, 0, 1, 0]))
            try:
                self.do_simulation(None, self.frame_skip)
            except MujocoException as e:
                print("Inside _reset_ee:", e)
        # curr_ee_pos = self.get_endeff_pos()
        # print(curr_ee_pos-new_mocap_pos_xy)

    def _reset_obj(self, type='ground'):
        obj = self._sample_realistic_obj(type=type, ee=self.get_endeff_pos(), mode='reset')

        self._set_obj_xyz(obj[0:3], obj_id=0)
        if self.two_obj:
            self._set_obj_xyz(obj[3:6], obj_id=1)
        
    def _reset_gripper(self, type='ground'):
        gripper = self._sample_realistic_gripper(type=type)
        try:
            self.do_simulation(gripper, self.frame_skip)
        except MujocoException as e:
            print("Inside _reset_gripper:", e)

    def _sample_realistic_ee(self, mode='goal'):
        if mode == 'reset' and self.fixed_reset is not None:
            return self.fixed_reset[0:3]
        if mode == 'goal' and self.fixed_goal is not None:
            return self.fixed_goal[0:3]

        ee = np.random.uniform(self.hand_space.low, self.hand_space.high)
        while self._ee_wall_collision(ee):
            ee = np.random.uniform(self.hand_space.low, self.hand_space.high)

        return ee

    def _sample_realistic_obj(self, type='ground', ee=None, mode='goal'):
        assert type in ['ground', 'stacked', 'air']

        if mode == 'reset' and self.fixed_reset is not None:
            if self.two_obj:
                return self.fixed_reset[3:9]
            else:
                return self.fixed_reset[3:6]

        if mode == 'goal' and self.fixed_goal is not None:
            if self.two_obj:
                return self.fixed_goal[3:9]
            else:
                return self.fixed_goal[3:6]

        # choose first obj to set
        if self.two_obj:
            obj_id = np.random.choice(2)
        else:
            obj_id = 0

        obj0, obj1 = None, None

        if type == 'ground':
            obj0 = self._sample_obj(obj_id=obj_id, type='ground')
            while self._ee_obj_collision(ee, obj0):
                obj0 = self._sample_obj(obj_id=obj_id, type='ground')

            if self.two_obj:
                obj_id = 1 - obj_id
                obj1 = self._sample_obj(obj_id=obj_id, type='ground')
                while self._ee_obj_collision(ee, obj1) or self._obj_obj_collision(obj0, obj1, mode=mode):
                    obj1 = self._sample_obj(obj_id=obj_id, other_obj=obj0)
        elif type == 'stacked':
            assert self.two_obj

            obj0 = self._sample_obj(obj_id=obj_id, type='ground')
            obj1_perfect = np.copy(obj0)
            obj1_perfect[2] = 0.045
            while self._ee_obj_collision(ee, obj0) or self._ee_obj_collision(ee, obj1_perfect):
                obj0 = self._sample_obj(obj_id=obj_id, type='ground')
                obj1_perfect = np.copy(obj0)
                obj1_perfect[2] = 0.045

            obj_id = 1 - obj_id
            obj1 = self._sample_obj(obj_id=obj_id, type='stacked', other_obj=obj0)
            while self._ee_obj_collision(ee, obj1):
                obj1 = self._sample_obj(obj_id=obj_id, type='stacked', other_obj=obj0)
        elif type == 'air':
            obj0 = self._sample_obj(obj_id=obj_id, type='air', ee=ee)

            if self.two_obj:
                obj_id = 1 - obj_id
                obj1 = self._sample_obj(obj_id=obj_id, type='ground')
                while self._ee_obj_collision(ee, obj1) or self._obj_obj_collision(obj0, obj1):
                    obj1 = self._sample_obj(obj_id=obj_id, type='ground')

        if self.two_obj:
            if obj_id == 1:
                return np.concatenate((obj0, obj1))
            else:
                return np.concatenate((obj1, obj0))
        else:
            return obj0
        
    def _sample_obj(self, obj_id=0, type='ground', ee=None, other_obj=None):
        if obj_id == 0:
            start_idx, end_idx = 3, 6
        elif obj_id == 1:
            start_idx, end_idx = 6, 9
        obj_xyz = None
        if type == 'ground':
            obj_xyz = np.random.uniform(
                self.hand_and_obj_space.low[start_idx:end_idx],
                self.hand_and_obj_space.high[start_idx:end_idx]
            )
            obj_xyz[2] = 0.015
            if 'wall' in self.structure:
                while (self.wall_radius[0]) \
                        < np.abs(obj_xyz[1] - self.wall_center[1]) \
                        <= (self.wall_radius[0] + self.obj_radius[0]):
                    obj_xyz = np.random.uniform(
                        self.hand_and_obj_space.low[start_idx:end_idx],
                        self.hand_and_obj_space.high[start_idx:end_idx]
                    )
                    obj_xyz[2] = 0.015
                if np.abs(obj_xyz[1] - self.wall_center[1]) < (self.wall_radius[0]):
                    obj_xyz[2] = 0.015 + self.wall_center[2] + self.wall_radius[1]
        elif type == 'stacked':
            obj_xyz = np.copy(other_obj)
            obj_xyz[1] += np.random.normal(0.0, 0.003)
            obj_xyz[2] = 0.045
        elif type == 'air':
            obj_xyz = np.copy(ee)
            obj_xyz[2] -= 0.02
        return obj_xyz

    def _sample_realistic_gripper(self, type='ground'):
        if type in ['ground', 'stacked']:
            if np.random.uniform() <= 0.25:
                return np.array([1])
            else:
                return np.array([-1])
        elif type == 'air':
            return np.array([1])

    def _obj_obj_collision(self, obj0, obj1, mode='goal'):
        # dist = np.linalg.norm(obj0 - obj1, ord=np.inf)
        # if mode == 'reset':
        #     return dist <= (self.obj_radius + self.obj_radius + 0.03) # extra room for robot to pick up both objs
        # else:
        #     return dist <= (self.obj_radius + self.obj_radius)
        diff = np.abs(obj0 - obj1)[-2:]
        if mode == 'reset':
            return np.all(diff < (self.obj_radius + self.obj_radius + np.array([0.03, 0])))
        else:
            return np.all(diff < (self.obj_radius + self.obj_radius))

    def _ee_obj_collision(self, ee, obj):
        # dist = np.linalg.norm(ee - obj, ord=np.inf)
        # return dist <= (self.ee_radius + self.obj_radius)
        diff = np.abs(ee - obj)[-2:]
        return np.all(diff < self.ee_radius + self.obj_radius)

    def _ee_wall_collision(self, ee):
        # dist = np.linalg.norm(ee - obj, ord=np.inf)
        # return dist <= (self.ee_radius + self.obj_radius)
        if 'wall' in self.structure:
            diff = np.abs(ee - self.wall_center)[-2:]
            return np.all(diff < self.ee_radius + self.wall_radius + np.array([0.015, 0.0]))
        else:
            return False

    """
    Multitask functions
    """
    def get_goal(self):
        return {
            'desired_goal': self._state_goal,
            'state_desired_goal': self._state_goal,
        }

    def set_goal(self, goal):
        self._state_goal = goal['state_desired_goal']
        self._set_goal_marker(self._state_goal)

    def sample_goals(self, batch_size):
        if self.num_goals_presampled > 0:
            if self._presampled_goals is None:
                self._presampled_goals = self._sample_realistic_goals(batch_size=self.num_goals_presampled)
            idx = np.random.randint(0, self.num_goals_presampled, batch_size)
            sampled_goals = {
                k: v[idx] for k, v in self._presampled_goals.items()
            }
        else:
            sampled_goals = self._sample_realistic_goals(batch_size=batch_size)
        return sampled_goals

    def compute_rewards(self, actions, obs, prev_obs=None, reward_type=None):
        if reward_type is None:
            reward_type = self.reward_type
        achieved_goals = obs['state_achieved_goal']
        desired_goals = obs['state_desired_goal']
        hand_pos = achieved_goals[:, :3]
        if self.two_obj:
            obj_pos = achieved_goals[:, 3:9]
        else:
            obj_pos = achieved_goals[:, 3:6]
        hand_goals = desired_goals[:, :3]
        if self.two_obj:
            obj_goals = desired_goals[:, 3:9]
        else:
            obj_goals = desired_goals[:, 3:6]

        hand_distances = np.linalg.norm(hand_goals - hand_pos, ord=self.norm_order, axis=1)
        obj_distances = np.linalg.norm(obj_goals - obj_pos,  ord=self.norm_order,axis=1)
        hand_and_obj_distances = hand_distances + obj_distances

        if reward_type == 'hand_distance':
            r = -hand_distances
        elif reward_type == 'vectorized_state_distance':
            r = -np.abs(achieved_goals - desired_goals)
        elif reward_type == 'telescoping_vectorized_state_distance':
            if prev_obs is None:
                return np.zeros((achieved_goals.shape))
            prev_achieved_goals = prev_obs['state_achieved_goal']
            prev_desired_goals = prev_obs['state_desired_goal']
            assert (desired_goals == prev_desired_goals).all()
            current_r = -np.abs(achieved_goals - desired_goals)
            old_r = -np.abs(prev_achieved_goals - prev_desired_goals)
            r = current_r - old_r
        elif reward_type == 'state_distance':
            r = -np.linalg.norm(achieved_goals - desired_goals, ord=self.norm_order, axis=1)
        elif reward_type == 'state_distance_wo_second_obj':
            achieved_goals_wo_second_obj = np.concatenate((achieved_goals[:, :6], achieved_goals[:, -1:]), axis=1)
            desired_goals_wo_second_obj = np.concatenate((desired_goals[:, :6], desired_goals[:, -1:]), axis=1)
            r = -np.linalg.norm(achieved_goals_wo_second_obj - desired_goals_wo_second_obj, ord=self.norm_order, axis=1)
        elif reward_type == 'hand_success':
            r = -(hand_distances > self.indicator_threshold).astype(float)
        elif reward_type == 'obj_distance':
            r = -obj_distances
        elif reward_type == 'obj_success':
            r = -(obj_distances > self.indicator_threshold).astype(float)
        elif reward_type == 'hand_and_obj_distance':
            r = -hand_and_obj_distances
        elif reward_type == 'hand_and_obj_success':
            r = -(
                hand_and_obj_distances < self.indicator_threshold
            ).astype(float)
        else:
            raise NotImplementedError("Invalid/no reward type.")
        return r

    def get_diagnostics(self, paths, prefix=''):
        statistics = OrderedDict()
        stat_names = [
            'hand_success',
            'obj_success',
            'hand_and_obj_success',
            'hand_distance',
            'obj_distance',
            'obj_distance0',
            'hand_and_obj_distance',
            'total_pickups',
            'obj_x0', 'obj_y0', 'obj_z0',
            'mujoco_exception_raised', 'flying_object',
        ]
        if self.two_obj:
            stat_names.append('obj_distance1')
            stat_names.append('obj_x1')
            stat_names.append('obj_y1')
            stat_names.append('obj_z1')
        for stat_name in stat_names:
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

    def get_env_state(self):
        base_state = super().get_env_state()
        if self._state_goal is not None:
            goal = self._state_goal.copy()
        else:
            goal = None
        return base_state, goal

    def set_env_state(self, state):
        base_state, goal = state
        super().set_env_state(base_state)
        self._state_goal = goal
        if goal is not None:
            self._set_goal_marker(goal)

    def generate_opposing_positions(self):
        raise NotImplementedError
        left_side_low = self.gripper_and_hand_and_obj_space.low.copy()
        left_side_high = self.gripper_and_hand_and_obj_space.high.copy()
        # object on left side of wall
        left_side_high[4] = .51
        # hand on right side of wall
        left_side_low[1] = .66
        left_side_high[5] = 0.0

        right_side_low = self.gripper_and_hand_and_obj_space.low.copy()
        right_side_high = self.gripper_and_hand_and_obj_space.high.copy()
        # object on left side of wall
        right_side_low[4] = .69
        # hand on right side of wall
        right_side_high[1] = .54
        right_side_high[5] = 0.0
        left_goals = np.random.uniform(
            left_side_low,
            left_side_high,
            size=(1, self.gripper_and_hand_and_obj_space.low.size),
        )
        right_goals = np.random.uniform(
            right_side_low,
            right_side_high,
            size=(1, self.gripper_and_hand_and_obj_space.low.size),
        )
        left_goal_dict = {
                'desired_goal': left_goals[0],
                'state_desired_goal': left_goals[0],
                'proprio_desired_goal': left_goals[:, :3][0],
                'is_obj_in_hand': False
        }
        right_goal_dict = {
                'desired_goal': right_goals[0],
                'state_desired_goal': right_goals[0],
                'proprio_desired_goal': right_goals[:, :3][0],
                'is_obj_in_hand': False
        }
        # make hand goal the start
        if np.random.random() > 0.5:
            return left_goal_dict, right_goal_dict
        else:
            return right_goal_dict, left_goal_dict

    def update_subgoals(self, latent_subgoals, latent_subgoals_noisy):
        self.subgoals = latent_subgoals

    def states_to_images(self, states):
        images = []
        for state in states:
            image = self.get_image_plt(draw_state=True, state=state)
            image = image.transpose()
            images.append(image)
        return np.array(images)

    def get_image_plt(self,
                      vals=None,
                      extent=None,
                      imsize=84,
                      draw_state=True, draw_goal=False, draw_subgoals=False,
                      state=None, goal=None, subgoals=None):
        if extent is None:
            x_bounds = np.array([self.hand_space.low[1] - 0.05, self.hand_space.high[1] + 0.05])
            y_bounds = np.array([self.hand_space.low[2] - 0.05, self.hand_space.high[2] + 0.05])
            self.vis_bounds = np.concatenate((x_bounds, y_bounds))
            extent = self.vis_bounds

        fig, ax = plt.subplots()
        ax.set_ylim(extent[2:4])
        ax.set_xlim(extent[0:2])
        ax.set_ylim(ax.get_ylim()[::])
        ax.set_xlim(ax.get_xlim()[::])
        DPI = fig.get_dpi()
        fig.set_size_inches(imsize / float(DPI), imsize / float(DPI))

        marker_factor = 0.50

        hand_low, hand_high = self.hand_space.low, self.hand_space.high
        ax.vlines(x=hand_low[1], ymin=hand_low[2], ymax=hand_high[2], linestyles='dotted', color='black')
        ax.hlines(y=hand_low[2], xmin=hand_low[1], xmax=hand_high[1], linestyles='dotted', color='black')
        ax.vlines(x=hand_high[1], ymin=hand_low[2], ymax=hand_high[2], linestyles='dotted', color='black')
        ax.hlines(y=hand_high[2], xmin=hand_low[1], xmax=hand_high[1], linestyles='dotted', color='black')

        ax.vlines(x=hand_low[1]- 0.05, ymin=hand_low[2]- 0.05, ymax=hand_high[2]+ 0.05, color='black')
        ax.hlines(y=hand_low[2]- 0.05, xmin=hand_low[1]- 0.05, xmax=hand_high[1]+ 0.05, color='black')
        ax.vlines(x=hand_high[1]+ 0.05, ymin=hand_low[2]- 0.05, ymax=hand_high[2]+ 0.05, color='black')
        ax.hlines(y=hand_high[2]+ 0.05, xmin=hand_low[1]- 0.05, xmax=hand_high[1]+ 0.05, color='black')

        if draw_state:
            if state is not None:
                hand_pos = state[1:3]
                obj_pos = state[4:6]
                if self.two_obj:
                    obj2_pos = state[7:9]
            else:
                hand_pos = self.get_endeff_pos()[1:3]
                obj_pos = self.get_obj_pos(obj_id=0)[1:3]
                if self.two_obj:
                    obj2_pos = self.get_obj_pos(obj_id=1)[1:3]
            hand = plt.Circle(hand_pos, 0.025 * marker_factor, color='green')
            ax.add_artist(hand)
            obj = plt.Circle(obj_pos, 0.025 * marker_factor, color='blue')
            ax.add_artist(obj)
            if self.two_obj:
                obj2 = plt.Circle(obj2_pos, 0.025 * marker_factor, color='red')
                ax.add_artist(obj2)
        if draw_goal:
            hand = plt.Circle(self._state_goal[1:3], 0.02 * marker_factor, color='yellowgreen')
            ax.add_artist(hand)
            obj = plt.Circle(self._state_goal[4:6], 0.02 * marker_factor, color='cyan')
            ax.add_artist(obj)
            if self.two_obj:
                obj2 = plt.Circle(self._state_goal[7:9], 0.02 * marker_factor, color='orange')
                ax.add_artist(obj2)
        if draw_subgoals:
            if self.subgoals is not None:
                subgoals = self.subgoals.reshape((-1, self.observation_space.spaces['state_observation'].low.size))
                for subgoal in subgoals[:1]:
                    hand = plt.Circle(subgoal[1:3], 0.015 * marker_factor, color='orange')
                    ax.add_artist(hand)
                    obj = plt.Circle(subgoal[4:6], 0.015 * marker_factor, color='cyan')
                    ax.add_artist(obj)
                    if self.two_obj:
                        obj2 = plt.Circle(subgoal[7:9], 0.015 * marker_factor, color='yellowgreen')
                        ax.add_artist(obj2)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        fig.subplots_adjust(bottom=0)
        fig.subplots_adjust(top=1)
        fig.subplots_adjust(right=1)
        fig.subplots_adjust(left=0)
        ax.axis('off')

        # ax.imshow(
        #     vals,
        #     extent=extent,
        #     cmap=plt.get_cmap('plasma'),
        #     interpolation='nearest',
        #     vmax=vmax,
        #     vmin=vmin,
        #     origin='bottom',  # <-- Important! By default top left is (0, 0)
        # )

        return self.plt_to_numpy(fig)

    def plt_to_numpy(self, fig):
        fig.canvas.draw()
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()
        return data

class SawyerPickAndPlaceEnvYZ(SawyerPickAndPlaceEnv):

    def __init__(
        self,
        x_axis=0.0,
        snap_obj_to_axis=False,
        set_vel_to_zero_in_snap_obj=False,
        *args,
        **kwargs
    ):
        self.quick_init(locals())
        self.x_axis = x_axis
        self.snap_obj_to_axis = snap_obj_to_axis
        self.set_vel_to_zero_in_snap_obj = set_vel_to_zero_in_snap_obj
        super().__init__(*args, **kwargs)
        pos_arrays = [
            self.hand_and_obj_space.low[:3],
            self.hand_and_obj_space.low[3:6],
            self.hand_and_obj_space.high[:3],
            self.hand_and_obj_space.high[3:6],

            self.gripper_and_hand_and_obj_space.low[0:3],
            self.gripper_and_hand_and_obj_space.low[3:6],
            self.gripper_and_hand_and_obj_space.high[0:3],
            self.gripper_and_hand_and_obj_space.high[3:6],

            self.hand_space.low[:3],
            self.hand_space.high[:3],

            self.obj_low,
            self.obj_high,

            self.hand_low,
            self.hand_high,
        ]
        for pos in pos_arrays:
            pos[0] = x_axis

        # for obj in [
        #     self.obj_low,
        #     self.obj_high,
        #     "",
        #     self.hand_low,
        #     self.hand_high,
        #     "",
        #     self.hand_and_obj_space.low,
        #     self.hand_and_obj_space.high,
        #     "",
        #     self.hand_space.low,
        #     self.hand_space.high,
        #     "",
        #     self.gripper_and_hand_and_obj_space.low,
        #     self.gripper_and_hand_and_obj_space.high
        # ]:
        #     print(obj)

        self.action_space = Box(
            np.array([-1, -1, -1]),
            np.array([1, 1, 1]),
            dtype=np.float32
        )

    def convert_2d_action(self, action):
        cur_x_pos = self.get_endeff_pos()[0]
        adjust_x = cur_x_pos - self.x_axis
        return np.r_[adjust_x, action]

    def _snap_obj_to_axis(self):
        if self.snap_obj_to_axis:
            new_obj_pos0 = self.get_obj_pos(obj_id=0)
            new_obj_pos0[0] = self.x_axis
            self._set_obj_xyz(new_obj_pos0, obj_id=0, set_vel_to_zero=self.set_vel_to_zero_in_snap_obj)
            if self.two_obj:
                new_obj_pos1 = self.get_obj_pos(obj_id=1)
                new_obj_pos1[0] = self.x_axis
                self._set_obj_xyz(new_obj_pos1, obj_id=1, set_vel_to_zero=self.set_vel_to_zero_in_snap_obj)

    def step(self, action):
        self._snap_obj_to_axis()
        action = self.convert_2d_action(action)
        return super().step(action)

    def set_to_goal(self, goal, **kwargs):
        super().set_to_goal(goal, **kwargs)
        self._snap_obj_to_axis()

    def expert_start(self):
        return np.array([
            0, .45, .15,
            0, .49, .02,
            0, .74, .02,
            0
        ])

    def expert_goal(self):
        return np.array([
            0, .70, .15,
            0, .71, .02,
            0, .51, .02,
            0
        ])

    def generate_expert_subgoals(self, num_subgoals):
        ob_and_goal = self._get_obs()
        ob = ob_and_goal['state_observation']
        goal = ob_and_goal['state_desired_goal']

        subgoals = []
        subgoals += [
            [
                0, .49, .10,
                0, .49, .09,
                0, .74, .02,
                0.05
            ],
            [
                0, .74, .1,
                0, .71, .02,
                0, .74, .02,
                0.00
            ],
            [
                0, .74, .1,
                0, .51, .08,
                0, .51, .02,
                0.00
            ],
            goal
        ]
        return np.array(subgoals)

def get_image_presampled_goals(image_env, num_goals_presampled):
    print("sampling image goals from pick_n_place env:", num_goals_presampled)
    t = time.time()
    image_env.reset()
    pickup_env = image_env.wrapped_env

    state_goals = np.zeros((num_goals_presampled, pickup_env.observation_space.spaces['state_desired_goal'].low.size))
    proprio_goals = np.zeros((num_goals_presampled, 3))
    image_goals = np.zeros((num_goals_presampled, image_env.image_length))
    for i in range(num_goals_presampled):
        type = pickup_env._sample_goal_type()
        ee = pickup_env._sample_realistic_ee()
        obj = pickup_env._sample_realistic_obj(type=type, ee=ee)
        gripper = pickup_env._sample_realistic_gripper(type=type)
        state_goals[i] = np.concatenate((ee, obj, gripper))

    pre_state = pickup_env.get_env_state()
    for i in range(num_goals_presampled):
        pickup_env.set_to_goal({'state_desired_goal': state_goals[i]})

        obs = image_env._get_obs()
        state_goals[i] = obs['state_achieved_goal']
        proprio_goals[i] = obs['proprio_achieved_goal']
        image_goals[i] = obs['image_achieved_goal']
    pickup_env.set_env_state(pre_state)

    print("total time:", time.time() - t)

    return {
        'desired_goal': image_goals,
        'image_desired_goal': image_goals,
        'state_desired_goal': state_goals,
        'proprio_desired_goal': proprio_goals,
    }

