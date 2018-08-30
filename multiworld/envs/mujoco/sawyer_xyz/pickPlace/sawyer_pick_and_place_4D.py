from collections import OrderedDict
import numpy as np
from gym.spaces import  Dict , Box


from multiworld.envs.env_util import get_stat_in_paths, \
    create_stats_ordered_dict, get_asset_full_path
from multiworld.core.multitask_env import MultitaskEnv
from multiworld.envs.mujoco.sawyer_xyz.base import SawyerXYZEnv

from  multiworld.envs.mujoco.sawyer_xyz.pickPlace.sawyer_pick_and_place import  SawyerPickPlaceEnv

class SawyerPickPlaceEnv_4D(SawyerPickPlaceEnv):

    def __init__(
            self,
            tasks,
        
            **kwargs
    ):


        self.quick_init(locals())

       
        SawyerPickPlaceEnv.__init__(self, tasks = tasks, **kwargs)

        self.action_space = Box(
            np.array([-1, -1, -1, -1, -1]),
            np.array([1, 1, 1, 1, 1]),
        )

  

    

    def step(self, action):

      

        self.set_xyzRot_action(action[:4])
      

        self.do_simulation([action[-1], -action[-1]])
        
        # The marker seems to get reset every time you do a simulation
        self._set_goal_marker(self._state_goal)



        self._set_objCOM_marker()

     
        ob = self._get_obs()
       
        reward , reachRew, reachDist, pickRew, placeRew , placingDist = self.compute_reward(action, ob)
        self.curr_path_length +=1

       
        #info = self._get_info()

        if self.curr_path_length == self.max_path_length:
            done = True
        else:
            done = False

        #print(pickRew)

        return ob, reward, done, { 'reachRew':reachRew, 'reachDist': reachDist, 'pickRew':pickRew, 'placeRew': placeRew, 'epRew' : reward, 'placingDist': placingDist}


   
