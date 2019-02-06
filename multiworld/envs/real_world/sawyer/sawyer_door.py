from collections import OrderedDict
from gym.spaces import Dict
import sawyer_control.envs.sawyer_door as sawyer_door
from multiworld.core.serializable import Serializable
from multiworld.core.multitask_env import MultitaskEnv
from multiworld.envs.env_util import create_stats_ordered_dict, get_stat_in_paths
import numpy as np

class SawyerDoorEnv(sawyer_door.SawyerDoorEnv, MultitaskEnv):
    ''' Must Wrap with Image Env to use!'''
    def __init__(self,
                 door_open_epsilon=2,
                 **kwargs
                ):
        self.door_open_epsilon=door_open_epsilon
        Serializable.quick_init(self, locals())
        sawyer_door.SawyerDoorEnv.__init__(self, **kwargs)
        self.observation_space = Dict([
            ('observation', self.observation_space),
            ('desired_goal', self.goal_space),
            ('achieved_goal', self.goal_space),
            ('state_observation', self.observation_space),
            ('state_desired_goal', self.goal_space),
            ('state_achieved_goal', self.goal_space),
        ])

    def step(self, action):
        self._act(action)
        observation = self._get_obs()
        reward = None
        info = self._get_info()
        done = False
        return observation, reward, done, info

    def _get_info(self):

        hand_distance = np.linalg.norm(self._state_goal[:3] - self._get_endeffector_pose())
        if self.use_dynamixel:
            if self.eval_mode == 'eval':
                relative_motor_pos = self._get_relative_motor_pos()
            else:
                relative_motor_pos = 0
            relative_abs_motor_angle_difference_from_goal = np.abs(self._state_goal[3] - relative_motor_pos)
            relative_abs_motor_angle_difference_from_reset =  np.abs(relative_motor_pos)
            relative_abs_motor_angle_difference_from_reset_indicator =  (relative_abs_motor_angle_difference_from_reset > self.door_open_epsilon).astype(float)
            return dict(
                hand_distance=hand_distance,
                relative_abs_motor_angle_difference_from_goal=relative_abs_motor_angle_difference_from_goal,
                relative_abs_motor_angle_difference_from_reset=relative_abs_motor_angle_difference_from_reset,
                relative_abs_motor_angle_difference_from_reset_indicator=relative_abs_motor_angle_difference_from_reset_indicator,
            )
        else:
            return dict(hand_distance=hand_distance)

    def compute_rewards(self, actions, obs):
        raise NotImplementedError('Use Image based reward')

    def _get_obs(self):
        if self.use_dynamixel:
            if self.eval_mode == 'eval':
                relative_motor_pos = self._get_relative_motor_pos()
            else:
                relative_motor_pos = 0
            achieved_goal = np.concatenate((self._get_endeffector_pose(), [relative_motor_pos]))
        else:
            achieved_goal = self._get_endeffector_pose()
        state_obs = super()._get_obs()
        return dict(
            observation=state_obs,
            desired_goal=self._state_goal,
            achieved_goal=achieved_goal,

            state_observation=state_obs,
            state_desired_goal=self._state_goal,
            state_achieved_goal=achieved_goal,
        )

    def reset(self):
        if self.use_dynamixel:
            super()._reset_robot_and_door()
        else:
            super()._reset_robot_and_door()
        goal = self.sample_goal()
        self._state_goal = goal['state_desired_goal']
        return self._get_obs()

    """
    Multitask functions
    """

    def get_goal(self):
        return {
            'desired_goal': self._state_goal,
            'state_desired_goal': self._state_goal,
        }

    def sample_goal(self):
        return MultitaskEnv.sample_goal(self)

    def sample_goals(self, batch_size):
        goals = super().sample_goals(batch_size)
        return {
            'desired_goal': goals,
            'state_desired_goal': goals,
        }

    def set_to_goal(self, goal):
        goal = goal['state_desired_goal']
        super().set_to_goal(goal)

    def set_goal(self, goal):
        self._state_goal = goal['state_desired_goal']

    def get_diagnostics(self, paths, prefix=''):
        statistics = OrderedDict()
        if self.use_dynamixel:
            stats = [
                'hand_distance',
                'relative_abs_motor_angle_difference_from_goal',
                'relative_abs_motor_angle_difference_from_reset',
                'relative_abs_motor_angle_difference_from_reset_indicator'
            ]
        else:
            stats = ['hand_distance']

        for stat_name in stats:
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

if __name__=="__main__":
    env = SawyerDoorEnv()
    env.reset()
