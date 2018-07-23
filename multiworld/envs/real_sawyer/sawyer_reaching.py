from collections import OrderedDict
import numpy as np

import time
from sawyer_control.sawyer_env_base import SawyerEnv
from rllab.spaces.box import Box
from sawyer_control.serializable import Serializable
from railrl.envs.multitask.multitask_env import MultitaskEnv
import cv2

JOINT_ANGLES_HIGH = np.array([
    1.70167993,
    1.04700017,
    3.0541791,
    2.61797006,
    3.05900002,
    2.09400001,
    3.05899961
])

JOINT_ANGLES_LOW = np.array([
    -1.70167995,
    -2.14700025,
    -3.0541801,
    -0.04995198,
    -3.05900015,
    -1.5708003,
    -3.05899989
])

JOINT_VEL_HIGH = 2*np.ones(7)
JOINT_VEL_LOW = -2*np.ones(7)

MAX_TORQUES = 0.5 * np.array([8, 7, 6, 5, 4, 3, 2])
JOINT_TORQUE_HIGH = MAX_TORQUES
JOINT_TORQUE_LOW = -1*MAX_TORQUES

JOINT_VALUE_HIGH = {
    'position': JOINT_ANGLES_HIGH,
    'velocity': JOINT_VEL_HIGH,
    'torque': JOINT_TORQUE_HIGH,
}
JOINT_VALUE_LOW = {
    'position': JOINT_ANGLES_LOW,
    'velocity': JOINT_VEL_LOW,
    'torque': JOINT_TORQUE_LOW,
}

END_EFFECTOR_POS_LOW = -1.2 * np.ones(3)
END_EFFECTOR_POS_HIGH = 1.2 *np.ones(3)

END_EFFECTOR_ANGLE_LOW = -1*np.ones(4)
END_EFFECTOR_ANGLE_HIGH = np.ones(4)

END_EFFECTOR_VALUE_LOW = {
    'position': END_EFFECTOR_POS_LOW,
    'angle': END_EFFECTOR_ANGLE_LOW,
}

END_EFFECTOR_VALUE_HIGH = {
    'position': END_EFFECTOR_POS_HIGH,
    'angle': END_EFFECTOR_ANGLE_HIGH,
}


class SawyerXYZReachingEnv(SawyerEnv):
    def __init__(self,
                 desired = None,
                 randomize_goal_on_reset=False,
                 **kwargs
                 ):
        Serializable.quick_init(self, locals())
        super().__init__(**kwargs)
        self.set_safety_box(
            pos_low=np.array([0.53, -.25, 0.35]),
            pos_high=np.array([0.675, 0.32, 0.5]),
            #torq_low=np.array([.3, -.25, .17]),
            #torq_high=np.array([0.675, .3, .55]),
            # torq_low=np.array([0, -.3, 0]),
            # torq_high=np.array([0.7, .3, 1]),
            #orq_low=np.array([0.37245887, -0.27259076, -0.05019562]),
            #torq_high=np.array([0.60924846, 0.68534386, 0.62648237]),
            torq_high=np.array([ 0.5637579,  0.5288438 ,  0.78652403]),
            torq_low=np.array([0.0, -0.38531035, 0.26874638]),
            ee_high = np.array([0.5637579, 0.4888438, 0.70652403]),
            ee_low = np.array([0.1, -0.33531035, 0.32874638]),

        )
        high = np.array([0.675, 0.32, 0.5])
        low = np.array([0.53, -.25, 0.35])
        if desired is None:
            self._randomize_desired_end_effector_pose()
        else:
            self.desired = desired
        self._randomize_goal_on_reset = randomize_goal_on_reset

    def reward(self):
        current = self._end_effector_pose()[:3]
        differences = self.desired - current
        reward = self.reward_function(differences)
        return reward


    def _get_random_ee_pose(self):
        high = self.pd_safety_box_high
        low = self.pd_safety_box_low
        dx = np.random.uniform(low[0], high[0])
        dy = np.random.uniform(low[1], high[1])
        dz = np.random.uniform(low[2], high[2])
        return np.array([dx, dy, dz])


    def _randomize_desired_end_effector_pose(self):
        self.desired = self._get_random_ee_pose()

    def reset(self):
        self.in_reset = True
        self._safe_move_to_neutral()
        self.in_reset = False
        if self._randomize_goal_on_reset:
            self._randomize_desired_end_effector_pose()
        return self._get_observation()




    def _get_statistics_from_paths(self, paths):
        statistics = OrderedDict()
        stat_prefix = 'Test'
        distances_from_target, final_position_distances, final_10_positions_distances = self._extract_experiment_info(paths)
        statistics.update(self._update_statistics_with_observation(
            distances_from_target,
            stat_prefix,
            'End Effector Distance from Target'
        ))

        statistics.update(self._update_statistics_with_observation(
            final_10_positions_distances,
            stat_prefix,
            'Last 10 Step End Effector Distance from Target'
        ))

        statistics.update(self._update_statistics_with_observation(
            final_position_distances,
            stat_prefix,
            'Final End Effector Distance from Target'
        ))

        return statistics

    def _extract_experiment_info(self, paths):
        obsSets = [path["next_observations"] for path in paths]
        positions = []
        desired_positions = []
        distances = []
        final_positions = []
        final_desired_positions = []
        final_10_positions = []
        final_10_desired_positions = []
        for obsSet in obsSets:
            for observation in obsSet:
                pos = np.array(observation['endpoint_pose'][:3])
                des = np.array(observation['goal_state'])
                distances.append(np.linalg.norm(pos - des))
                positions.append(pos)
                desired_positions.append(des)
            final_10_positions = positions[len(positions)-10:]
            final_10_desired_positions = desired_positions[len(desired_positions)-10:]
            final_positions.append(positions[-1])
            final_desired_positions.append(desired_positions[-1])

        distances = np.array(distances)
        final_positions = np.array(final_positions)
        final_desired_positions = np.array(final_desired_positions)
        final_10_positions = np.array(final_10_positions)
        final_10_desired_positions = np.array(final_10_desired_positions)
        final_10_distances = np.linalg.norm(final_10_positions - final_10_desired_positions, axis=1)
        final_position_distances = np.linalg.norm(final_positions - final_desired_positions, axis=1)
        return [distances, final_position_distances, final_10_distances]

class SawyerXYReachingEnv(SawyerEnv):
    '''
    Position control only
    fixed z-value, only move around in plane
    '''
    def __init__(self,
                 desired = None,
                 randomize_goal_on_reset=False,
                 z=0.3,
                 **kwargs
                 ):
        Serializable.quick_init(self, locals())
        super().__init__(**kwargs)
        self.action_mode = 'position'
        self.z = z
        if desired is None:
            self._randomize_desired_end_effector_pose()
        else:
            self.desired = desired
            desired[2] = self.z
        self._randomize_goal_on_reset = randomize_goal_on_reset


    #TODO: ADD position control option
    def _joint_act(self, action):
        ee_pos = self._end_effector_pose()
        if self.relative_pos_control:
            target_ee_pos = (ee_pos[:3] + action)
        else:
            target_ee_pos = action
        target_ee_pos = np.clip(target_ee_pos, self.pd_safety_box_low, self.pd_safety_box_high)
        target_ee_pos = np.concatenate((target_ee_pos, ee_pos[3:]))
        target_ee_pos[2] = self.z
        angles = self.request_ik_angles(target_ee_pos, self._joint_angles())
        self.send_angle_action(angles)

    def reward(self):
        current = self._end_effector_pose()[:2]
        differences = self.desired[:2] - current
        reward = self.reward_function(differences)
        return reward

    def _set_observation_space(self):
        lows = np.hstack((
            JOINT_VALUE_LOW['position'],
            JOINT_VALUE_LOW['velocity'],
            JOINT_VALUE_LOW['torque'],
            END_EFFECTOR_VALUE_LOW['position'],
            END_EFFECTOR_VALUE_LOW['angle'],
            END_EFFECTOR_VALUE_LOW['position'],
        ))

        highs = np.hstack((
            JOINT_VALUE_HIGH['position'],
            JOINT_VALUE_HIGH['velocity'],
            JOINT_VALUE_HIGH['torque'],
            END_EFFECTOR_VALUE_HIGH['position'],
            END_EFFECTOR_VALUE_HIGH['angle'],
            END_EFFECTOR_VALUE_HIGH['position'],
        ))

        self._observation_space = Box(lows, highs)


    def _get_random_ee_pose(self):
        high = self.pd_safety_box_high
        low = self.pd_safety_box_low
        dx = np.random.uniform(low[0], high[0])
        dy = np.random.uniform(low[1], high[1])
        return np.array([dx, dy, self.z])


    def _randomize_desired_end_effector_pose(self):
        self.desired = self._get_random_ee_pose()

    def reset(self):
        self.in_reset = True
        self._safe_move_to_neutral()
        self.in_reset = False
        if self._randomize_goal_on_reset:
            self._randomize_desired_end_effector_pose()
        return self._get_observation()

    def _get_statistics_from_paths(self, paths):
        statistics = OrderedDict()
        stat_prefix = 'Test'
        distances_from_target, final_position_distances, final_10_positions_distances = self._extract_experiment_info(paths)
        statistics.update(self._update_statistics_with_observation(
            distances_from_target,
            stat_prefix,
            'End Effector Distance from Target'
        ))

        statistics.update(self._update_statistics_with_observation(
            final_10_positions_distances,
            stat_prefix,
            'Last 10 Step End Effector Distance from Target'
        ))

        statistics.update(self._update_statistics_with_observation(
            final_position_distances,
            stat_prefix,
            'Final End Effector Distance from Target'
        ))

        return statistics

    def _extract_experiment_info(self, paths):
        obsSets = [path["next_observations"] for path in paths]
        positions = []
        desired_positions = []
        distances = []
        final_positions = []
        final_desired_positions = []
        final_10_positions = []
        final_10_desired_positions = []
        for obsSet in obsSets:
            for observation in obsSet:
                pos = np.array(observation[21:24])
                des = np.array(observation[28:31])
                distances.append(np.linalg.norm(pos - des))
                positions.append(pos)
                desired_positions.append(des)
            final_10_positions = positions[len(positions)-10:]
            final_10_desired_positions = desired_positions[len(desired_positions)-10:]
            final_positions.append(positions[-1])
            final_desired_positions.append(desired_positions[-1])

        distances = np.array(distances)
        final_positions = np.array(final_positions)
        final_desired_positions = np.array(final_desired_positions)
        final_10_positions = np.array(final_10_positions)
        final_10_desired_positions = np.array(final_10_desired_positions)
        final_10_distances = np.linalg.norm(final_10_positions - final_10_desired_positions, axis=1)
        final_position_distances = np.linalg.norm(final_positions - final_desired_positions, axis=1)
        return [distances, final_position_distances, final_10_distances]


class SawyerXYZReachingMultitaskEnv(SawyerEnv, MultitaskEnv):
    def __init__(self,
                 desired = None,
                 randomize_goal_on_reset=False,
                 **kwargs
                 ):
        Serializable.quick_init(self, locals())
        SawyerEnv.__init__(self, **kwargs)
        MultitaskEnv.__init__(self, **kwargs)
        # if desired is None:
        #     self._randomize_desired_end_effector_pose()
        # else:
        #     self.desired = desired
        self._randomize_goal_on_reset = randomize_goal_on_reset
        high = self.pd_safety_box_high
        low = self.pd_safety_box_low
        self.goal_space = Box(low, high)

    @property
    def goal_dim(self):
        return 3

    def convert_obs_to_goals(self, obs):
        return obs['goal_state'] # TODO: check this!

    def sample_goals(self, batch_size):
        return np.vstack([self.goal_space.sample() for i in range(batch_size)])

    def set_goal(self, goal):

        #self.MultitaskEnv.set_goal(goal)
        MultitaskEnv.set_goal(self, goal)
        # For VAE: actually need to move the arm to this position
        self.desired = goal


    def reward(self):
        current = self._end_effector_pose()[:3]
        differences = self.desired - current
        reward = self.reward_function(differences)
        return reward


    def _randomize_desired_end_effector_pose(self):
        #self.desired = np.random.uniform(self.safety_box_lows, self.safety_box_highs, size=(1, 3))[0]
        assert False # Don't randomize goal inside reset! In MultitaskEnv this is done by set_goal
        high = self.pd_safety_box_high
        low = self.pd_safety_box_low
        dx = np.random.uniform(low[0], high[0])
        dy = np.random.uniform(low[1], high[1])
        dz = np.random.uniform(low[2], high[2])
        self.desired =  np.array([dx, dy, dz])

    def compute_her_reward_np(self, observation, action, next_observation):
        return -np.linalg.norm(next_observation['endpoint_pose'][:3] - next_observation['goal_state'])

    def reset(self):
        self.in_reset = True
        self._safe_move_to_neutral()
        self.in_reset = False
        return self._get_observation()


    def step(self, action):
        self._act(action)
        observation = self._get_observation()
        reward = self.reward() * self.reward_magnitude
        done = False
        info = {}
        info['goal'] = self.desired.copy()
        return observation, reward, done, info

    def _get_statistics_from_paths(self, paths):
        statistics = OrderedDict()
        stat_prefix = 'Test'
        distances_from_target, final_position_distances = self._extract_experiment_info(paths)
        statistics.update(self._update_statistics_with_observation(
            distances_from_target,
            stat_prefix,
            'End Effector Distance from Target'
        ))
        statistics.update(self._update_statistics_with_observation(
            final_position_distances,
            stat_prefix,
            'Final End Effector Distance from Target'
        ))
        return statistics

    def _extract_experiment_info(self, paths):
        obsSets = [path["next_observations"] for path in paths]
        goalSet = [path['goals'][0] for path in paths]
        positions = []
        desired_positions = []
        distances = []
        final_positions = []
        final_desired_positions = []
        for obsSet, goal in zip(obsSets, goalSet):
            for observation in obsSet:
                pos = np.array(observation[21:24])
                distances.append(np.linalg.norm(pos - goal))
                positions.append(pos)
                desired_positions.append(goal)
            final_positions.append(positions[-1])
            final_desired_positions.append(desired_positions[-1])

        distances = np.array(distances)
        final_positions = np.array(final_positions)
        final_desired_positions = np.array(final_desired_positions)
        final_position_distances = np.linalg.norm(final_positions - final_desired_positions, axis=1)
        return [distances, final_position_distances]




class SawyerXYZReachingImgMultitaskEnv(SawyerEnv, MultitaskEnv):
    def __init__(self,
                 desired = None,
                 randomize_goal_on_reset=False,
                 reprsentation_size = 16,
                 **kwargs
                 ):
        Serializable.quick_init(self, locals())
        SawyerEnv.__init__(self, **kwargs)
        MultitaskEnv.__init__(self, **kwargs)
        self.desired = np.zeros((3))
        #vae safety box:
        self.set_safety_box(
            pos_low=np.array([0.53, -.25, 0.35]),
            pos_high=np.array([0.65, 0.32, 0.5]),
            torq_low=np.array([.3, -.25, .17]),
            torq_high=np.array([0.65, .3, .55]),
        )
        high = np.array([0.65, 0.32, 0.5])
        low = np.array([0.53, -.25, 0.35])
        self.goal_space = Box(low, high)

        # self.goals = np.load('/home/mdalal/Documents/railrl-private/ee_positions.npy')
        # self.imgs = np.load('/home/mdalal/Documents/railrl-private/images.npy')
        # self.goal_idx = None
    @property
    def goal_dim(self):
        return 3

    def convert_obs_to_goals(self, obs):
        return obs['goal_state']

    def sample_goals(self, batch_size):
        #sample a random set of goals from goal dict
        return np.vstack([self.goal_space.sample() for i in range(batch_size)])

    def set_goal(self, goal):
        MultitaskEnv.set_goal(self, goal)
        # For VAE: actually need to move the arm to this position
        #instead of moving the arm to the position
        # have a goal to image dict
        self.thresh = False
        print('Setting desired joint position:', goal)
        self._joint_act(goal - self._end_effector_pose()[:3])
        self.desired = goal
        self.thresh = True
        # return self.imgs[self.goal_idx]

    def reward(self):
        current = self._end_effector_pose()[:3]
        differences = self.desired - current
        reward = self.reward_function(differences)
        return reward



    def reset(self):
        self.in_reset = True
        self._safe_move_to_neutral()
        self.in_reset = False
        observation = self.get_image()
        return observation

    def step(self, action):
        self._act(action)
        observation = self._get_observation()
        reward = self.reward() * self.reward_magnitude
        done = False
        info = {}
        pos = np.array(observation['endpoint_pos'][:3])
        info['distance_to_goal'] = np.linalg.norm(pos - self.desired)
        return observation, reward, done, info


    def _get_statistics_from_paths(self, paths):
        statistics = OrderedDict()
        stat_prefix = 'Test'
        #distances_from_target = np.array([path['distance_to_goal'] for path in paths]).flatten()
        distances_from_target = []
        final_position_distances = []
        for path in paths:
            distances_from_target += [p['distance_to_goal'] for p in path['env_infos']]
            final_position_distances += [[p['distance_to_goal'] for p in path['env_infos']][-1]]
        final_position_distances = np.array(final_position_distances)
        distances_from_target = np.array(distances_from_target)
        statistics.update(self._update_statistics_with_observation(
            distances_from_target,
            stat_prefix,
            'End Effector Distance from Target'
        ))

        statistics.update(self._update_statistics_with_observation(
            final_position_distances,
            stat_prefix,
            'Final End Effector Distance from Target'
        ))

        return statistics


class SawyerXYPushingImgMultitaskEnv(SawyerEnv, MultitaskEnv):

    def __init__(self,
                 desired = None,
                 randomize_goal_on_reset=False,
                 #z=0.3,
                 pause_on_reset=False,
                 **kwargs
                 ):
        Serializable.quick_init(self, locals())
        SawyerEnv.__init__(self, **kwargs)
        MultitaskEnv.__init__(self, **kwargs)
        self.desired = np.zeros((4))
        self.action_mode = 'position'
        self.pause_on_reset = pause_on_reset
        #set pushing safety box here
        #goals are push the object to a spot, then reach a different position
        self.pushing_box_high_obj = np.array([0.66, 0.1]) # for the same = 0.02265
        self.pushing_box_low_obj = np.array([0.45, -0.1])
        self.pushing_box_high_ee = np.array([0.76, 0.145])
        self.pushing_box_low_ee = np.array([0.45, -0.12])
        self.pushing_box_low = np.array([0.44, -0.12])
        self.pushing_box_high = np.array([0.76, 0.145])
        high = np.concatenate((self.pushing_box_high_obj, self.pushing_box_high_ee)) #make sure to concatenate with itself
        low = np.concatenate((self.pushing_box_low_obj, self.pushing_box_low_ee))
        self.goal_space = Box(low, high)
        self.img_observation = True
        self.z = 0.23128
        self.img_observation = True
        self.reset_position = np.array([0.4499366724, 0.02265634, 0.23128])
        #self.set_safety_box(ee_low=np.array([0.405, -0.13, 0.23128]), ee_high=np.array([0.92, 0.156, 0.23128]))


    @property
    def goal_dim(self):
        return 4 #xy for object position, xy for end effector position
    def convert_obs_to_goals(self, obs):
        return obs[:, 21:24] # TODO: check this!
    def sample_goals(self, batch_size):
        return np.vstack([self.goal_space.sample() for i in range(batch_size)])

    def set_goal(self, goal,eval=True):
        print('pause')
        input()
        if eval:
            print('setting goal')
            obj_goal = np.concatenate((goal[:2], [self.z]))
            ee_goal = np.concatenate((goal[2:4], [self.z]))
            MultitaskEnv.set_goal(self, goal)
            #PAUSE FOR INPUT:
            self.thresh=False
            self._joint_act(obj_goal-self._end_effector_pose()[:3])
            input()
            self._joint_act(ee_goal - self._end_effector_pose()[:3])
            print('setting ee')
            input()
            #PAUSE FOR INPUT
            #PLACE OBJECT AT GOAL POSITION
            #MOVES ROBOT ARM TO GOAL POSITION:

            self.desired = goal
            self.thresh = True


    def get_image(self):
        temp = self.request_image()
        #update the get image in server. get an 84 x 84 x 3
        #reshape to 84 x 84 x 3
        temp = np.array(temp)
        temp = temp.reshape(84, 84, 3)
        img = temp[:40, 12:63]
        img = cv2.resize(img, (0, 0), fx=1.64705882, fy=2.1)
        observation = img
        observation = img / 255.0
        observation = observation.transpose()
        observation = observation.flatten()
        return observation

    def reward(self):
        '''
        this is isnt used by the multitask env at all
        :return:
        '''
        current = self._end_effector_pose()[:3]
        # differences = self.desired - current
        # reward = self.reward_function(differences)
        return current


    def _randomize_desired_end_effector_pose(self):
        #self.desired = np.random.uniform(self.safety_box_lows, self.safety_box_highs, size=(1, 3))[0]
        assert False # Don't randomize goal inside reset! In MultitaskEnv this is done by set_goal
        high = self.ee_safety_box_high
        low = self.ee_safety_box_low
        dx = np.random.uniform(low[0], high[0])
        dy = np.random.uniform(low[1], high[1])
        dz = np.random.uniform(low[2], high[2])
        self.desired =  np.array([dx, dy, dz])

    def compute_her_reward_np(self, observation, action, next_observation):
        '''
        this shouldn't be used either
        :param observation:
        :param action:
        :param next_observation:
        :param goal:
        :return:
        '''
        return 0

    def reset(self):
        self.in_reset = True
        self.thresh = False
        self._joint_act(self.reset_position  - self._end_effector_pose()[:3])
        self.in_reset = False
        self.thresh = True
        if self.pause_on_reset:
            #PAUSE AND MOVE OBJECT TO ITS RESET POSITION
            print('in reset')
            input()
        if self.img_observation:
            observation = self.get_image()
        else:
            observation = self._get_observation()
        return observation

    def _joint_act(self, action):
        ee_pos = self._end_effector_pose()
        if self.relative_pos_control:
            target_ee_pos = (ee_pos[:3] + action)
        else:
            target_ee_pos = action
        target_ee_pos[2] = self.z
        target_ee_pos[:2] = np.clip(target_ee_pos[:2], self.pushing_box_low, self.pushing_box_high)
        target_ee_pos = np.concatenate((target_ee_pos, ee_pos[3:]))
        angles = self.request_ik_angles(target_ee_pos, self._joint_angles())
        self.send_angle_action(angles)



    def step(self, action):
        self._act(action)
        observation = self._get_observation()
        reward = self.reward() * self.reward_magnitude
        done = False
        info = {}
        return observation, reward, done, info


    def _get_statistics_from_paths(self, paths):
        return OrderedDict()
