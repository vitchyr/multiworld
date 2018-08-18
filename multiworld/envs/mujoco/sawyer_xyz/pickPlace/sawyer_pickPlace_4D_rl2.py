from collections import OrderedDict
import numpy as np
from gym.spaces import  Dict , Box


from multiworld.envs.env_util import get_stat_in_paths, \
    create_stats_ordered_dict, get_asset_full_path
from multiworld.core.multitask_env import MultitaskEnv
from multiworld.envs.mujoco.sawyer_xyz.base import SawyerXYZEnv

from  multiworld.envs.mujoco.sawyer_xyz.pickPlace.sawyer_pick_and_place_4D import  SawyerPickPlaceEnv_4D



class SawyerPickPlaceEnv_4D_rl2(SawyerPickPlaceEnv_4D):

    def __init__(
            self,
            tasks,
        
            **kwargs
    ):


        self.quick_init(locals())

       
        SawyerPickPlaceEnv_4D.__init__(self, tasks = tasks, **kwargs)

        self.observation_space = self.hand_and_obj_space
        



    def _get_obs(self):
        hand = self.get_endeff_pos()
        objPos = self.get_body_com("obj")
      
        flat_obs = np.concatenate((hand, objPos))


      

        return flat_obs


    


    def reset_trial(self):

        self.sim.reset()       
        self._reset_hand()
        task = self.sample_task()
        
        self._state_goal = np.array(task['goal'])        
        self.obj_init_pos = task['obj_init_pos']        
       
        self.heightTarget = 0.06
        self._set_goal_marker(self._state_goal)
        self._set_obj_xyz(self.obj_init_pos)

        self.curr_path_length = 0
        self.pickCompleted = False

        self.maxPlacingDist = np.linalg.norm(np.array([self.obj_init_pos[0], self.obj_init_pos[1], self.heightTarget]) - np.array(self._state_goal)) + self.heightTarget

        if self.viewer is not None:
            self.viewer_setup()

        return self._get_obs()


    def reset(self):

        self.sim.reset()       
        self._reset_hand()
     
        self._set_obj_xyz(self.obj_init_pos)

        self.curr_path_length = 0
        self.pickCompleted = False

        if self.viewer is not None:
            self.viewer_setup()

        return self._get_obs()




  