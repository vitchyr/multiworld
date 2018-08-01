from collections import OrderedDict
import numpy as np
from gym.spaces import Box, Dict

from multiworld.envs.env_util import get_stat_in_paths, \
    create_stats_ordered_dict, get_asset_full_path
from multiworld.core.multitask_env import MultitaskEnv
from multiworld.envs.mujoco.sawyer_xyz.base import SawyerXYZEnv

import pickle

class SawyerPickEnv(SawyerXYZEnv):
    def __init__(
            self,
            obj_low=None,
            obj_high=None,

            reward_type='hand_and_obj_distance',
            indicator_threshold=0.06,

            obj_init_pos=(0, 0.6, 0.02),

            goal_idx =None,
           
            goal_low=None,
            goal_high=None,

            hide_goal_markers=False,

            **kwargs
    ):
        self.quick_init(locals())
       # MultitaskEnv.__init__(self)
        SawyerXYZEnv.__init__(
            self,
            model_name=self.model_name,
            **kwargs
        )
        if obj_low is None:
            obj_low = self.hand_low
        if obj_high is None:
            obj_high = self.hand_high

        if goal_low is None:
            goal_low = np.hstack((self.hand_low, obj_low))
        if goal_high is None:
            goal_high = np.hstack((self.hand_high, obj_high))


        self.max_path_length = 100

        self.reward_type = reward_type
        self.indicator_threshold = indicator_threshold

        self.obj_init_pos = np.array(obj_init_pos)

        self._goal_idx = goal_idx

        self.hide_goal_markers = hide_goal_markers

        self.action_space = Box(
            np.array([-1, -1, -1, -1]),
            np.array([1, 1, 1, 1]),
        dtype = np.float32)
        self.hand_and_obj_space = Box(
            np.hstack((self.hand_low, obj_low)),
            np.hstack((self.hand_high, obj_high)),
        dtype = np.float32)
      
        self.goals = pickle.load(open("/home/russellm/multiworld/multiworld/envs/goals/sawyer_pick_goals_file1.pkl", "rb"))

        self.observation_space = self.hand_and_obj_space
            
       
        #goal is not part of observation space, specified in the dict only to define 
        #self.goal_space. Goal not passed into the network
       

        self.heightTarget = 0.1
        

    @property
    def model_name(self):
        return get_asset_full_path('sawyer_xyz/sawyer_pick_and_place.xml')

    def viewer_setup(self):
        pass
        # self.viewer.cam.trackbodyid = 0
        # self.viewer.cam.lookat[0] = 0
        # self.viewer.cam.lookat[1] = 1.0
        # self.viewer.cam.lookat[2] = 0.5
        # self.viewer.cam.distance = 0.3
        # self.viewer.cam.elevation = -45
        # self.viewer.cam.azimuth = 270
        # self.viewer.cam.trackbodyid = -1

    def  log_diagnostics(self, paths, prefix):
        pass



    def step(self, action):

     
        self.set_xyz_action(action[:3])


       
        
        self.do_simulation([action[-1], -action[-1]])
        # The marker seems to get reset every time you do a simulation
        #self._set_goal_marker(self._state_goal)
        ob = self._get_obs()
       

        reward , pickRew = self.compute_rewards(action)
        self.curr_path_length +=1

       
        #info = self._get_info()

        if self.curr_path_length == self.max_path_length:
            done = True
        else:
            done = False
        return ob, reward, done, {'pickRew':pickRew}

    def _get_obs(self):
        e = self.get_endeff_pos()
        b = self.get_obj_pos()
        flat_obs = np.concatenate((e, b))

       

        return flat_obs

    def get_obj_pos(self):
        return self.data.get_body_xpos('obj').copy()

    def _set_goal_marker(self, goal):
        """
        This should be use ONLY for visualization. Use self._state_goal for
        logging, learning, etc.
        """
        self.data.site_xpos[self.model.site_name2id('goal')] = (
            goal[:3]
        )
       
        if self.hide_goal_markers:
            self.data.site_xpos[self.model.site_name2id('goal'), 2] = (
                -1000
            )
           

    def _set_obj_xyz(self, pos):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()

        qpos[9:12] = pos.copy()
        qvel[9:15] = 0
        self.set_state(qpos, qvel)

    def sample_goals(self, num_goals):

      
        #return np.random.choice(np.array(range(len(self.goals))), num_goals)   (subsampling)

        return np.array(range(num_goals))
       

    #@overrides
    def reset(self, reset_args = None):
        self.sim.reset()
        ob = self.reset_model(reset_args = reset_args)
        if self.viewer is not None:
            self.viewer_setup()
        return ob

    def reset_model(self, reset_args = None):

        #print(reset_args)
        goal_idx = reset_args
        if goal_idx is not None:
            self._goal_idx = goal_idx
        elif self._goal_idx is None:
            self._goal_idx = np.random.randint(1)

        self._reset_hand()
        
        xy_pos = self.goals[self._goal_idx]
        #self._set_goal_marker(self._state_goal)
        new_block_pos = np.array([xy_pos[0], xy_pos[1], 0.02])
        self._set_obj_xyz(new_block_pos)

        self.curr_path_length = 0

        return self._get_obs()
      

    def _reset_hand(self):
        for _ in range(10):
            self.data.set_mocap_pos('mocap', np.array([0, 0.5, 0.05]))
            self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
            self.do_simulation(None, self.frame_skip)

    

    def get_site_pos(self, siteName):
        _id = self.model.site_names.index(siteName)
        return self.data.site_xpos[_id].copy()





    def compute_rewards(self, actions):
           
        rightFinger, leftFinger = self.get_site_pos('rightEndEffector'), self.get_site_pos('leftEndEffector')
       
        heightTarget = self.heightTarget
       
        objPos = self.get_body_com("obj")
        #print(objPos)
      

        
        fingerCOM = (rightFinger + leftFinger)/2


        graspDist = np.linalg.norm(objPos - fingerCOM)
        graspRew = -graspDist


        def graspAttained():
            if graspDist <0.1:
                return True

            else:
                return False

      
        def pickReward():

           
       
            if (objPos[2]> 0.025) and graspAttained():
                
                return 10* min(heightTarget, objPos[2])
         
            else:
                return 0

       

        pickRew = pickReward()
       
        reward = graspRew + pickRew 


       
        return [reward, pickRew] 
        #returned in a list because that's how compute_reward in multiTask.env expects it

   

    def get_diagnostics(self, paths, prefix=''):
        statistics = OrderedDict()
        # for stat_name in [
        #     'hand_distance',
        #     'obj_distance',
        #     'hand_and_obj_distance',
        #     'touch_distance',
        #     'hand_success',
        #     'obj_success',
        #     'hand_and_obj_success',
        #     'touch_success',
        # ]:
        #     stat_name = stat_name
        #     stat = get_stat_in_paths(paths, 'env_infos', stat_name)
        #     statistics.update(create_stats_ordered_dict(
        #         '%s%s' % (prefix, stat_name),
        #         stat,
        #         always_show_all_stats=True,
        #     ))
        #     statistics.update(create_stats_ordered_dict(
        #         'Final %s%s' % (prefix, stat_name),
        #         [s[-1] for s in stat],
        #         always_show_all_stats=True,
        #     ))
        return statistics


    #required by rllab parallel sampler
    def terminate(self):
        """
        Clean up operation,
        """
        pass


   
