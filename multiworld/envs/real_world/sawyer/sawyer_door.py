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
        self.set_mode('eval')

    def step(self, action):
        self._act(action)
        observation = self._get_obs()
        reward = None
        info = self._get_info()
        done = False
        return observation, reward, done, info

    def _get_info(self):
        hand_distance = np.linalg.norm(self._state_goal[:3] - self._get_endeffector_pose())
        return dict(hand_distance=hand_distance)

    def compute_rewards(self, actions, obs):
        raise NotImplementedError('Use Image based reward')

    def _get_obs(self):
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
        if not self.reset_free or self.eval_mode=='eval':
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

    def set_mode(self, mode):
        self.eval_mode = mode

if __name__=="__main__":
    env = SawyerDoorEnv()
    env.reset()
