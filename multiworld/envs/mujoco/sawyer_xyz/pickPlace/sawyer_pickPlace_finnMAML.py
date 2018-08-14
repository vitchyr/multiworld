from collections import OrderedDict
import numpy as np
from gym.spaces import Box, Dict

from multiworld.envs.env_util import get_stat_in_paths, \
    create_stats_ordered_dict, get_asset_full_path
from multiworld.core.multitask_env import MultitaskEnv
from multiworld.envs.mujoco.sawyer_xyz.base import SawyerXYZEnv

from  multiworld.envs.mujoco.sawyer_xyz.pickPlace.sawyer_pick_and_place import  SawyerPickPlaceEnv
import pickle


class SawyerPickPlace_finnMAMLEnv(SawyerPickPlaceEnv):

    def __init__(
            self,
        
            **kwargs
    ):


        self.quick_init(locals())

        self._goal_idx = None


        #tasks = pickle.load(open('/home/russellm/multiworld/multiworld/envs/goals/pickPlace_20X20_6_8.pkl', 'rb'))
        tasks = pickle.load(open('/root/code/multiworld/multiworld/envs/goals/pickPlace_60X30.pkl', 'rb'))
        #tasks = pickle.load(open('/root/code/multiworld/multiworld/envs/goals/pickPlace_20X20_6_8.pkl', 'rb'))

        SawyerPickPlaceEnv.__init__(self, tasks = tasks, **kwargs)

        self.observation_space = self.hand_and_obj_space
        



    def _get_obs(self):
        hand = self.get_endeff_pos()
        objPos = self.get_body_com("obj")
      
        flat_obs = np.concatenate((hand, objPos))


      

        return flat_obs



    def sample_goals(self, num_goals):

        #assert num_goals == len(self.tasks)
        #no subsampling

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
        
        self._state_goal = task['goal']
        self.obj_init_pos = task['obj_init_pos']
        
       
        self.heightTarget = 0.06

        self._set_goal_marker(self._state_goal)

        self._set_obj_xyz(self.obj_init_pos)

        self.curr_path_length = 0
        self.pickCompleted = False

        

        self.maxPlacingDist = np.linalg.norm(np.array([self.obj_init_pos[0], self.obj_init_pos[1], self.heightTarget]) - np.array(self._state_goal)) + self.heightTarget
        #Can try changing this
        return self._get_obs()



    def log_diagnostics(self, paths, prefix):
        pass

    
    #required by rllab parallel sampler
    def terminate(self):
        """
        Clean up operation,
        """
        pass
        
    

