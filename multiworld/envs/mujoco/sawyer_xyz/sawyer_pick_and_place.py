from collections import OrderedDict
import numpy as np
from gym.spaces import Box, Dict

from multiworld.envs.env_util import get_stat_in_paths, \
    create_stats_ordered_dict, get_asset_full_path
from multiworld.core.multitask_env import MultitaskEnv
from multiworld.envs.mujoco.sawyer_xyz.base import SawyerXYZEnv
from multiworld.envs.mujoco.cameras import sawyer_pick_and_place_camera


class SawyerPickAndPlaceEnv(MultitaskEnv, SawyerXYZEnv):
    def __init__(
            self,
            obj_low=None,
            obj_high=None,

            reward_type='hand_and_obj_distance',
            indicator_threshold=0.06,

            obj_init_positions=((0, 0.6, 0),),
            random_init=False,

            fix_goal=False,
            fixed_goal=(0.15, 0.6, 0.055, -0.15, 0.6),
            goal_low=None,
            goal_high=None,
            reset_free=False,
            hard_goals=False,

            hide_goal_markers=False,
            oracle_reset_prob=0.0,
            presampled_goals=None,
            num_goals_presampled=1000,
            norm_order=2,
            p_obj_in_hand=.75,
            structure='wall',

            **kwargs
    ):
        self.quick_init(locals())
        # self.obj_id[0] for first object, self.obj_id[1] for second object.
        self.obj_id = {
            0: '',
            1: '2',
        }

        self.structure = structure
        self.hard_goals = hard_goals
        MultitaskEnv.__init__(self)
        SawyerXYZEnv.__init__(
            self,
            model_name=self.model_name,
            **kwargs
        )
        self.norm_order = norm_order
        if obj_low is None:
            obj_low = self.hand_low
            obj_low[2] = 0.0
        if obj_high is None:
            obj_high = self.hand_high
        self.obj_low = obj_low
        self.obj_high = obj_high
        if goal_low is None:
            goal_low = np.hstack((self.hand_low, obj_low))
        if goal_high is None:
            goal_high = np.hstack((self.hand_high, obj_high))

        self.reward_type = reward_type
        self.random_init = random_init
        self.p_obj_in_hand = p_obj_in_hand
        self.indicator_threshold = indicator_threshold

        self.obj_init_z = obj_init_positions[0][2]
        self.obj_init_positions = np.array(obj_init_positions)
        self.last_obj_pos = self.obj_init_positions[0]

        self.subgoals = None
        self.fix_goal = fix_goal
        self.fixed_goal = np.array(fixed_goal)
        self._state_goal = None
        self.reset_free = reset_free
        self.train_oracle_reset_prob = oracle_reset_prob
        self.oracle_reset_prob = self.train_oracle_reset_prob

        self.hide_goal_markers = hide_goal_markers

        self.action_space = Box(
            np.array([-1, -1, -1, -1]),
            np.array([1, 1, 1, 1]),
            dtype=np.float32
        )
        self.hand_and_obj_space = Box(
            np.hstack((self.hand_low, obj_low, obj_low)),
            np.hstack((self.hand_high, obj_high, obj_high)),
            dtype=np.float32
        )
        self.hand_space = Box(
            self.hand_low,
            self.hand_high,
            dtype=np.float32
        )
        self.gripper_and_hand_and_obj_space = Box(
            np.hstack((self.hand_low, obj_low, obj_low, [0.0])),
            np.hstack((self.hand_high, obj_high, obj_high, [0.05])),
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
        self.hand_reset_pos = np.array([0, .6, .2])

        if presampled_goals is not None:
            self._presampled_goals = presampled_goals
            self.num_goals_presampled = len(list(self._presampled_goals.values)[0])
        else:
            # presampled_goals will be created when sample_goal is first called
            self._presampled_goals = None
            self.num_goals_presampled = num_goals_presampled
        self.picked_up_object = False
        self.train_pickups = 0
        self.eval_pickups = 0
        self.cur_mode = 'train'
        self.reset()

    @property
    def model_name(self):
        if self.structure == 'wall':
            return get_asset_full_path('sawyer_xyz/sawyer_pick_and_place_wall.xml')
        else:
            return get_asset_full_path('sawyer_xyz/sawyer_pick_and_place.xml')

    def train(self):
        self.oracle_reset_prob = self.train_oracle_reset_prob
        self.cur_mode = 'train'

    def eval(self):
        self.oracle_reset_prob = 0.0
        self.cur_mode = 'eval'

    def mode(self, mode):
        if 'train' in mode:
            self.train()
        else:
            self.eval()

    def viewer_setup(self):
        sawyer_pick_and_place_camera(self.viewer.cam)

    def _handle_out_of_bounds(self, obj_id):
        new_obj_pos = self.get_obj_pos(obj_id)
        # if the object is out of bounds and not in the air, move it back
        new_obj_pos = np.clip(
            new_obj_pos,
            self.obj_low,
            self.obj_high
        )
        self._set_obj_xyz(new_obj_pos, obj_id)

    def step(self, action):
        prev_ob = self._get_obs()
        self.set_xyz_action(action[:3])
        self.do_simulation(action[3:])
        self._handle_out_of_bounds(obj_id=0)
        self._handle_out_of_bounds(obj_id=1)
        if (
            self.get_obj_pos(obj_id=0)[2] > .07 or
            self.get_obj_pos(obj_id=1)[2] > .07
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
        reward = self.compute_reward(action, ob, prev_obs=prev_ob)
        info = self._get_info()
        done = False
        return ob, reward, done, info

    def _get_obs(self):
        e = self.get_endeff_pos()
        b0 = self.get_obj_pos(obj_id=0)
        b1 = self.get_obj_pos(obj_id=1)
        b0 = np.clip(
            b0,
            self.obj_low,
            self.obj_high
        )
        b1 = np.clip(
            b1,
            self.obj_low,
            self.obj_high
        )

        gripper = self.get_gripper_pos()
        flat_obs = np.concatenate((e, b0, b1))
        flat_obs_with_gripper = np.concatenate((e, b0, b1, gripper))
        hand_goal = self._state_goal[:3]
        return dict(
            observation=flat_obs_with_gripper,
            desired_goal=self._state_goal,
            achieved_goal=flat_obs_with_gripper,
            state_observation=flat_obs_with_gripper,
            state_desired_goal=self._state_goal,
            state_achieved_goal=flat_obs_with_gripper,
            proprio_observation=e,
            proprio_achieved_goal=e,
            proprio_desired_goal=hand_goal,
        )

    def _get_info(self):
        hand_goal = self._state_goal[:3]
        obj_goal0 = self._state_goal[3:6]
        obj_goal1 = self._state_goal[6:9]
        hand_distance = np.linalg.norm(hand_goal - self.get_endeff_pos(), ord=self.norm_order,)
        obj_distance0 = np.linalg.norm(obj_goal0 - self.get_obj_pos(obj_id=0), ord=self.norm_order,)
        obj_distance1 = np.linalg.norm(obj_goal1 - self.get_obj_pos(obj_id=1), ord=self.norm_order,)
        obj_distance = obj_distance0 + obj_distance1
        return dict(
            hand_distance=hand_distance,
            obj_distance0=obj_distance0,
            obj_distance1=obj_distance1,
            hand_and_obj_distance=hand_distance+obj_distance,
            hand_success=float(hand_distance < self.indicator_threshold),
            obj_success=float(obj_distance < self.indicator_threshold),
            hand_and_obj_success=float(
                hand_distance+obj_distance < self.indicator_threshold
            ),
            total_pickups=self.train_pickups if self.cur_mode == 'train' else self.eval_pickups,
        )

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

    def _set_obj_xyz(self, pos, obj_id):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        obj_name = 'objjoint' + self.obj_id[obj_id]
        qpos_obj_start, qpos_obj_end = self.sim.model.get_joint_qpos_addr(obj_name)
        qvel_obj_start, qvel_obj_end = self.sim.model.get_joint_qvel_addr(obj_name)
        # the qpos is 7 dimensions. It might be x, y, z, quad. Ignore quad for now
        qpos[qpos_obj_start:qpos_obj_start + 3] = pos.copy()
        qvel[qvel_obj_start:qvel_obj_end] = 0
        self.set_state(qpos, qvel)


    def reset_model(self):
        self._reset_hand()

        if self.random_init:
            obj_goals = self.generate_uncorrected_env_goals(
                1, p_obj_in_hand=0
            )['state_desired_goal'][0][3:9]
            self._set_obj_xyz(obj_goals[0:3], obj_id=0)
            self._set_obj_xyz(obj_goals[3:6], obj_id=1)
            self.set_to_goal(
                {'state_desired_goal': self.expert_start()}
            )

        else:
            raise NotImplementedError

        if self.oracle_reset_prob > np.random.random():
            uncorrected_goal = self.generate_uncorrected_env_goals(1)
            self.set_to_goal(
                {'state_desired_goal': uncorrected_goal['state_desired_goal'][0]},
                is_obj_in_hand=bool(uncorrected_goal['is_obj_in_hand'][0][0])
            )

        self.set_goal(self.sample_goal())
        self.set_goal({'state_desired_goal': self.expert_goal()})

        self._set_goal_marker(self._state_goal)
        if self.hard_goals:
            reset_pos_dict, goal_dict = self.generate_opposing_positions()
            self.set_to_goal(reset_pos_dict)
            self.set_goal(goal_dict)

        self.picked_up_object = False
        return self._get_obs()

    def _reset_hand(self):
        for _ in range(10):
            self.data.set_mocap_pos('mocap', self.hand_reset_pos)
            self.data.set_mocap_quat('mocap', np.array([0, 0, 1, 0]))
            self.do_simulation(None, self.frame_skip)

    def set_to_goal(self, goal, is_obj_in_hand=False):
        """
        This function can fail due to mocap imprecision or impossible object
        positions.
        """
        state_goal = goal['state_desired_goal']
        hand_goal = state_goal[:3]
        for _ in range(30):
            self.data.set_mocap_pos('mocap', hand_goal)
            self.data.set_mocap_quat('mocap', np.array([0, 0, 1, 0]))
            self.do_simulation(np.array([-1]))
        # error = self.data.get_site_xpos('endeffector') - hand_goal
        corrected_obj_pos0 = state_goal[3:6] #+ error
        corrected_obj_pos0[2] = max(corrected_obj_pos0[2], self.obj_init_z)
        self._set_obj_xyz(corrected_obj_pos0, obj_id=0)
        corrected_obj_pos1 = state_goal[6:9] #+ error
        corrected_obj_pos1[2] = max(corrected_obj_pos1[2], self.obj_init_z)
        self._set_obj_xyz(corrected_obj_pos1, obj_id=1)

        if is_obj_in_hand:
            action = np.array(1)
        else:
            action = np.array(1 - 2 * np.random.choice(2))

        for _ in range(10):
            self.do_simulation(action)
        self.sim.forward()
        self._handle_out_of_bounds(obj_id=0)
        self._handle_out_of_bounds(obj_id=1)
        for _ in range(10):
            self.do_simulation(action)
        self.sim.forward()


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
        if self._presampled_goals is None:
            self._presampled_goals = \
                    corrected_state_goals(
                        self,
                        self.generate_uncorrected_env_goals(
                            self.num_goals_presampled
                        )
                    )
        idx = np.random.randint(0, self.num_goals_presampled, batch_size)
        sampled_goals = {
            k: v[idx] for k, v in self._presampled_goals.items()
        }
        return sampled_goals


    def compute_rewards(self, actions, obs, prev_obs=None, reward_type=None):
        if reward_type is None:
            reward_type = self.reward_type
        achieved_goals = obs['state_achieved_goal']
        desired_goals = obs['state_desired_goal']
        hand_pos = achieved_goals[:, :3]
        obj_pos = achieved_goals[:, 3:9]
        hand_goals = desired_goals[:, :3]
        obj_goals = desired_goals[:, 3:9]

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
        for stat_name in [
            'hand_success',
            'obj_success',
            'hand_and_obj_success',
            'hand_distance',
            'obj_distance',
            'obj_distance0',
            'obj_distance1',
            'hand_and_obj_distance',
            'total_pickups',
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

    def get_env_state(self):
        base_state = super().get_env_state()
        goal = self._state_goal.copy()
        return base_state, goal

    def set_env_state(self, state):
        base_state, goal = state
        super().set_env_state(base_state)
        self._state_goal = goal
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





    def generate_uncorrected_env_goals(self, num_goals, p_obj_in_hand=None):
        """
        Due to small errors in mocap, moving to a specified hand position may be
        slightly off. This is an issue when the object must be placed into a given
        hand goal since high precision is needed. The solution used is to try and
        set to the goal manually and then take whatever goal the hand and object
        end up in as the "corrected" goal. The downside to this is that it's not
        possible to call set_to_goal with the corrected goal as input as mocap
        errors make it impossible to rereate the exact same hand position.

        The return of this function should be passed into
        corrected_image_env_goals or corrected_state_env_goals
        """
        if p_obj_in_hand is None:
            p_obj_in_hand = self.p_obj_in_hand
        is_obj_in_hand = np.zeros((num_goals, 1))
        if self.fix_goal:
            goals = np.repeat(self.fixed_goal.copy()[None], num_goals, 0)
        else:
            goals = np.random.uniform(
                self.gripper_and_hand_and_obj_space.low,
                self.gripper_and_hand_and_obj_space.high,
                size=(num_goals, self.gripper_and_hand_and_obj_space.low.size),
            )
            num_objs_in_hand = int(num_goals * p_obj_in_hand)
            if num_goals == 1:
                num_objs_in_hand = int(np.random.random() < p_obj_in_hand)

            # Put object in hand
            for idx in range(num_objs_in_hand):
                obj_start_idx = 3 + 3 * np.random.choice(2)
                goals[idx, obj_start_idx:obj_start_idx + 3] = goals[idx, :3].copy()
                # center the object inside the gripper
                goals[idx, obj_start_idx + 1] -= 0.01
                goals[idx, obj_start_idx + 2] += 0.01
                is_obj_in_hand[idx] = 1

            # Put object one the table (not floating)
            goals[num_objs_in_hand:, 5] = self.obj_init_z
            goals[num_objs_in_hand:, 8] = self.obj_init_z
            if self.structure == 'wall':
                for goal in goals:
                    if goal[4] > 0.55 and goal[4] < 0.65:
                        goal[5] = 0.05
                    if goal[7] > 0.55 and goal[7] < 0.65:
                        goal[8] = 0.05
                    if goal[1] > 0.55 and goal[1] < 0.65 and goal[2] < 0.03:
                        goal[2] = 0.05
            return {
                'desired_goal': goals,
                'state_desired_goal': goals,
                'proprio_desired_goal': goals[:, :3],
                'is_obj_in_hand': is_obj_in_hand
            }

    def update_subgoals(self, subgoals):
        self.subgoals = subgoals

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
        import matplotlib
        matplotlib.use('agg')
        import matplotlib.pyplot as plt
        if extent is None:
            x_bounds = np.array([self.hand_space.low[1] - 0.05, self.hand_space.high[1] + 0.05])
            y_bounds = np.array([self.hand_space.low[2] - 0.05, self.hand_space.high[2] + 0.05])
            self.vis_bounds = np.concatenate((x_bounds, y_bounds))
            extent = self.vis_bounds

        fig, ax = plt.subplots()
        ax.set_ylim(extent[2:4])
        ax.set_xlim(extent[0:2])
        ax.set_ylim(ax.get_ylim()[::-1])
        ax.set_xlim(ax.get_xlim()[::-1])
        DPI = fig.get_dpi()
        fig.set_size_inches(imsize / float(DPI), imsize / float(DPI))

        marker_factor = 0.50

        hand_low, hand_high = self.hand_space.low, self.hand_space.high
        ax.vlines(x=hand_low[1], ymin=hand_low[2], ymax=hand_high[2], linestyles='dotted', color='black')
        ax.hlines(y=hand_low[2], xmin=hand_low[1], xmax=hand_high[1], linestyles='dotted', color='black')
        ax.vlines(x=hand_high[1], ymin=hand_low[2], ymax=hand_high[2], linestyles='dotted', color='black')
        ax.hlines(y=hand_high[2], xmin=hand_low[1], xmax=hand_high[1], linestyles='dotted', color='black')

        # puck_low, puck_high = self.puck_space.low, self.puck_space.high
        puck_low, puck_high = hand_low, hand_high
        # TODO
        ax.vlines(x=puck_low[1], ymin=puck_low[2], ymax=puck_high[2], linestyles='dotted', color='black')
        ax.hlines(y=puck_low[2], xmin=puck_low[1], xmax=puck_high[1], linestyles='dotted', color='black')
        ax.vlines(x=puck_high[1], ymin=puck_low[2], ymax=puck_high[2], linestyles='dotted', color='black')
        ax.hlines(y=puck_high[2], xmin=puck_low[1], xmax=puck_high[1], linestyles='dotted', color='black')

        if draw_state:
            if state is not None:
                hand_pos = state[1:3]
                puck_pos = state[4:6]
                puck2_pos = state[7:9]
            else:
                hand_pos = self.get_endeff_pos()[1:3]
                puck_pos = self.get_obj_pos(obj_id=0)[1:3]
                puck2_pos = self.get_obj_pos(obj_id=1)[1:3]
            hand = plt.Circle(hand_pos, 0.025 * marker_factor, color='red')
            ax.add_artist(hand)
            puck = plt.Circle(puck_pos, 0.025 * marker_factor, color='blue')
            ax.add_artist(puck)
            puck2 = plt.Circle(puck2_pos, 0.025 * marker_factor, color='green')
            ax.add_artist(puck2)
        if draw_goal:
            hand = plt.Circle(self._state_goal[1:3], 0.03 * marker_factor, color='orange')
            ax.add_artist(hand)
            puck = plt.Circle(self._state_goal[4:6], 0.03 * marker_factor, color='cyan')
            ax.add_artist(puck)
            puck2 = plt.Circle(self._state_goal[7:9], 0.03 * marker_factor, color='yellowgreen')
            ax.add_artist(puck2)
        if draw_subgoals:
            if self.subgoals is not None:
                subgoals = self.subgoals.reshape((-1, self.observation_space.spaces['state_observation'].low.size))
                print(subgoals)
                for subgoal in subgoals[:1]:
                    hand = plt.Circle(subgoal[1:3], 0.015 * marker_factor, color='orange')
                    ax.add_artist(hand)
                    puck = plt.Circle(subgoal[4:6], 0.015 * marker_factor, color='cyan')
                    ax.add_artist(puck)
                    puck2 = plt.Circle(subgoal[7:9], 0.015 * marker_factor, color='yellowgreen')
                    ax.add_artist(puck2)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        fig.subplots_adjust(bottom=0)
        fig.subplots_adjust(top=1)
        fig.subplots_adjust(right=1)
        fig.subplots_adjust(left=0)

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
        import matplotlib
        matplotlib.use('agg')
        import matplotlib.pyplot as plt
        fig.canvas.draw()
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        # data = data.reshape(fig.canvas.get_width_height() + (3,))
        plt.close()
        return data

class SawyerPickAndPlaceEnvYZ(SawyerPickAndPlaceEnv):

    def __init__(
        self,
        x_axis=0.0,
        *args,
        **kwargs
    ):
        self.quick_init(locals())
        self.x_axis = x_axis
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

            self.hand_space.high[:3],
            self.hand_space.low[:3],
        ]
        for pos in pos_arrays:
            pos[0] = x_axis

        self.action_space = Box(
            np.array([-1, -1, -1]),
            np.array([1, 1, 1]),
            dtype=np.float32
        )
        self.hand_reset_pos = np.array([x_axis, .6, .2])

    def convert_2d_action(self, action):
        cur_x_pos = self.get_endeff_pos()[0]
        adjust_x = self.x_axis - cur_x_pos
        return np.r_[adjust_x, action]

    def _snap_obj_to_axis(self):
        new_obj_pos0 = self.get_obj_pos(obj_id=0)
        new_obj_pos0[0] = self.x_axis
        self._set_obj_xyz(new_obj_pos0, obj_id=0)
        new_obj_pos1 = self.get_obj_pos(obj_id=1)
        new_obj_pos1[0] = self.x_axis
        self._set_obj_xyz(new_obj_pos1, obj_id=1)

    def step(self, action):
        self._snap_obj_to_axis()
        action = self.convert_2d_action(action)
        return super().step(action)

    def set_to_goal(self, goal, **kwargs):
        super().set_to_goal(goal, **kwargs)
        self._snap_obj_to_axis()

    def reset(self):
        super().reset()
        self._state_goal[0] = 0.0
        self._state_goal[3] = 0.0
        self._state_goal[6] = 0.0
        return self._get_obs()

    def _get_obs(self):
        e = self.get_endeff_pos()
        b0 = self.get_obj_pos(obj_id=0)
        b1 = self.get_obj_pos(obj_id=1)
        b0 = np.clip(
            b0,
            self.obj_low,
            self.obj_high
        )
        b1 = np.clip(
            b1,
            self.obj_low,
            self.obj_high
        )

        e[0] = 0.0
        b0[0] = 0.0
        b1[0] = 0.0
        gripper = self.get_gripper_pos()
        flat_obs = np.concatenate((e, b0, b1))
        flat_obs_with_gripper = np.concatenate((e, b0, b1, gripper))
        hand_goal = self._state_goal[:3]

        return dict(
            observation=flat_obs_with_gripper,
            desired_goal=self._state_goal,
            achieved_goal=flat_obs_with_gripper,
            state_observation=flat_obs_with_gripper,
            state_desired_goal=self._state_goal,
            state_achieved_goal=flat_obs_with_gripper,
            proprio_observation=e,
            proprio_achieved_goal=e,
            proprio_desired_goal=hand_goal,
        )

    def _get_info(self):
        e = self.get_endeff_pos()
        b0 = self.get_obj_pos(obj_id=0)
        b1 = self.get_obj_pos(obj_id=1)
        b0 = np.clip(
            b0,
            self.obj_low,
            self.obj_high
        )
        b1 = np.clip(
            b1,
            self.obj_low,
            self.obj_high
        )

        e[0] = 0.0
        b0[0] = 0.0
        b1[0] = 0.0
        hand_goal = self._state_goal[:3]
        obj_goal0 = self._state_goal[3:6]
        obj_goal1 = self._state_goal[6:9]
        hand_distance = np.linalg.norm(hand_goal - e,  ord=self.norm_order)
        obj_distance0 = np.linalg.norm(obj_goal0 - b0, ord=self.norm_order)
        obj_distance1 = np.linalg.norm(obj_goal1 - b1, ord=self.norm_order)
        obj_distance = obj_distance0 + obj_distance1
        return dict(
            hand_distance=hand_distance,
            obj_distance=obj_distance,
            obj_distance0=obj_distance0,
            obj_distance1=obj_distance1,
            hand_and_obj_distance=hand_distance+obj_distance,
            hand_success=float(hand_distance < self.indicator_threshold),
            obj_success=float(obj_distance < self.indicator_threshold),
            hand_and_obj_success=float(
                hand_distance+obj_distance < self.indicator_threshold
            ),
            total_pickups=self.train_pickups if self.cur_mode == 'train' else self.eval_pickups,
        )

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

class SawyerPushOneD(SawyerPickAndPlaceEnvYZ):
    def __init__(
        self,
        *args,
        **kwargs
    ):
        super().__init__(
            p_obj_in_hand=0.0,
            structure='none',
            *args,
            **kwargs
        )

    def step(self, action):
        action[2] = 1
        super().step(action)

def corrected_state_goals(pickup_env, pickup_env_goals):
    pickup_env._state_goal = np.zeros(
        pickup_env.observation_space.spaces['state_desired_goal'].low.size
    )
    goals = pickup_env_goals.copy()
    num_goals = len(list(goals.values())[0])
    for idx in range(num_goals):
        if idx % 100 == 0:
            print(idx)
        pickup_env.set_to_goal(
            {'state_desired_goal': goals['state_desired_goal'][idx]},
            is_obj_in_hand=bool(pickup_env_goals['is_obj_in_hand'][idx][0])
        )
        corrected_state_goal = pickup_env._get_obs()['achieved_goal']
        corrected_proprio_goal = pickup_env._get_obs()['proprio_achieved_goal']

        goals['desired_goal'][idx] = corrected_state_goal
        goals['proprio_desired_goal'][idx] = corrected_proprio_goal
        goals['state_desired_goal'][idx] = corrected_state_goal
    return goals

def corrected_image_env_goals(image_env, pickup_env_goals):
    """
    This isn't as easy as setting to the corrected since mocap will fail to
    move to the exact position, and the object will fail to stay in the hand.
    """
    pickup_env = image_env.wrapped_env
    image_env.wrapped_env._state_goal = np.zeros(
        pickup_env.observation_space.spaces['state_desired_goal'].low.size
    )
    goals = pickup_env_goals.copy()

    num_goals = len(list(goals.values())[0])
    goals = dict(
        image_desired_goal=np.zeros((num_goals, image_env.image_length)),
        desired_goal=np.zeros((num_goals, image_env.image_length)),
        state_desired_goal=np.zeros(
            (num_goals,
             pickup_env.observation_space.spaces['state_desired_goal'].low.size
            )
        ),
        proprio_desired_goal=np.zeros((num_goals, 3))
    )
    for idx in range(num_goals):
        if idx % 100 == 0:
            print(idx)
        image_env.set_to_goal(
            {'state_desired_goal': pickup_env_goals['state_desired_goal'][idx]},
            is_obj_in_hand=bool(pickup_env_goals['is_obj_in_hand'][idx][0])
        )
        corrected_state_goal = image_env._get_obs()['state_achieved_goal']
        corrected_proprio_goal = image_env._get_obs()['proprio_achieved_goal']
        corrected_image_goal = image_env._get_obs()['image_achieved_goal']

        goals['image_desired_goal'][idx] = corrected_image_goal
        goals['desired_goal'][idx] = corrected_image_goal
        goals['state_desired_goal'][idx] = corrected_state_goal
        goals['proprio_desired_goal'][idx] = corrected_proprio_goal
    return goals

def get_image_presampled_goals(image_env, num_presampled_goals):
    image_env.reset()
    pickup_env = image_env.wrapped_env
    image_env_goals = corrected_image_env_goals(
        image_env,
        pickup_env.generate_uncorrected_env_goals(num_presampled_goals)
    )
    return image_env_goals

