from collections import OrderedDict
import numpy as np
from gym.spaces import Box, Dict

from multiworld.envs.env_util import get_stat_in_paths, \
    create_stats_ordered_dict, get_asset_full_path
from multiworld.core.multitask_env import MultitaskEnv
from multiworld.envs.mujoco.sawyer_xyz.base import SawyerXYZEnv

from  multiworld.envs.mujoco.sawyer_xyz.door.sawyer_door_open import  SawyerDoorOpenEnv
import pickle


class SawyerDoorOpen_finnMAMLEnv(SawyerDoorOpenEnv):

    def __init__(
            self,
        
            **kwargs
    ):


        self.quick_init(locals())

        self._goal_idx = None
        tasks = pickle.load(open('/home/russellm/multiworld/multiworld/envs/goals/doorOpening_60X20X20.pkl', 'rb'))



        #self.tasks = pickle.load(open(tasksFile, 'rb'))

        SawyerDoorOpenEnv.__init__(self, tasks = tasks, **kwargs)

        self.observation_space = self.hand_and_door_space
          


    def _get_obs(self):
        e = self.get_endeff_pos()
        b = self.get_site_pos('doorGraspPoint')
        flat_obs = np.concatenate((e, b))

        return flat_obs




    def sample_goals(self, num_goals):

        return np.array(range(num_goals))
       

    #@overrides
    def reset(self, reset_args = None):
        self.sim.reset()
        ob = self.reset_model(reset_args = reset_args)
        if self.viewer is not None:
            self.viewer_setup()
        return ob

    def reset_model(self, reset_args = None):

      
        goal_idx = reset_args
        if goal_idx is not None:
            self._goal_idx = goal_idx
        elif self._goal_idx is None:
            self._goal_idx = np.random.randint(1)

        self._reset_hand()


        task = self.tasks[self._goal_idx]

        self._state_goal = task['goalAngle'][0]
        self.door_init_pos = task['door_init_pos']



        self._set_goal_marker()

        self._set_door_xyz(self.door_init_pos)

        self.curr_path_length = 0
      
       
        return self._get_obs()
        
    

