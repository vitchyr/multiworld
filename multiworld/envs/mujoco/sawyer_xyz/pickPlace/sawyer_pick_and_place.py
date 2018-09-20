from collections import OrderedDict
import numpy as np
from gym.spaces import Box, Dict

from multiworld.envs.env_util import get_stat_in_paths, \
    create_stats_ordered_dict, get_asset_full_path
from multiworld.core.multitask_env import MultitaskEnv
from multiworld.envs.mujoco.sawyer_xyz.push.sawyer_push import SawyerPushEnv


class SawyerPickPlaceEnv( SawyerPushEnv):
    def __init__(
            self,
            tasks = [{'goal': np.array([0, 0.7, 0.02]), 'height': 0.06, 'obj_init_pos':np.array([0, 0.6, 0.02])}] , 
            liftThresh = 0.04,
            rewMode = 'orig',
            **kwargs
    ):
  
        self.quick_init(locals())
        
        SawyerPushEnv.__init__(
            self,
            tasks = tasks,
            **kwargs
        )
        self.rewMode = rewMode
        self.heightTarget = self.objHeight + liftThresh
        self.action_space = Box(
            np.array([-1, -1, -1, -1]),
            np.array([1, 1, 1, 1]),
        )

    def _reset_hand(self):

        for _ in range(10):
            self.data.set_mocap_pos('mocap', self.hand_init_pos)
            self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
            self.do_simulation([-1,1], self.frame_skip)

    
    def step(self, action):

        
        self.set_xyz_action(action[:3])
        self.do_simulation([action[-1], -action[-1]])
        
      
        self._set_goal_marker(self._state_goal)
        ob = self._get_obs()
       

        reward , reachRew, reachDist, pickRew, placeRew , placingDist = self.compute_reward(action, ob)
        self.curr_path_length +=1


        if self.curr_path_length == self.max_path_length:
            done = True
        else:
            done = False


        return ob, reward, done, { 'reachRew':reachRew, 'reachDist': reachDist, 'pickRew':pickRew, 'placeRew': placeRew, 'epRew' : reward, 'placingDist': placingDist} 
                                    #'epObs': ob['state_observation']}



    def change_task(self, task):

       
        self._state_goal = self.adjust_goalPos(task['goal'])
        self._set_goal_marker(self._state_goal)

        self.obj_init_pos = self.adjust_initObjPos(task['obj_init_pos'])

        self.maxPlacingDist = np.linalg.norm(np.array([self.obj_init_pos[0], self.obj_init_pos[1], self.heightTarget]) - np.array(self._state_goal)) + self.heightTarget


    def compute_reward(self, actions, obs):
           
         
        if isinstance(obs, dict):
           
            obs = obs['state_observation']

        objPos = obs[3:6]
        rightFinger, leftFinger = self.get_site_pos('rightEndEffector'), self.get_site_pos('leftEndEffector')
        fingerCOM  =  (rightFinger + leftFinger)/2
       
        heightTarget = self.heightTarget
        placingGoal = self._state_goal
      
        graspDist = np.linalg.norm(objPos - fingerCOM)
        placingDist = np.linalg.norm(objPos - placingGoal)
      

        def reachReward():

            graspRew = -graspDist
            #incentive to close fingers when graspDist is small
            if graspDist < 0.02:

                graspRew = -graspDist + max(actions[-1],0)/50
            return graspRew , graspDist

        def pickCompletionCriteria():

            tolerance = 0.01

            if objPos[2] >= (heightTarget - tolerance):

                return True
            else:
                return False

        if pickCompletionCriteria():
            self.pickCompleted = True


        def grasped():

            sensorData = self.data.sensordata

            return (sensorData[0]>0) and (sensorData[1]>0)


        def objDropped():

            return (objPos[2] < (self.objHeight + 0.005)) and (placingDist >0.02) and (graspDist > 0.02) 
            # Object on the ground, far away from the goal, and from the gripper
            #Can tweak the margin limits


        def orig_pickReward():
            
            hScale = 50

            if self.pickCompleted and not(objDropped()):
                return hScale*heightTarget
       
            elif (objPos[2]> (self.objHeight + 0.005)) and (graspDist < 0.1):


                
                return hScale* min(heightTarget, objPos[2])
         
            else:
                return 0



        def general_pickReward():
            
            hScale = 50

            if self.pickCompleted and grasped():
                return hScale*heightTarget
       
            elif (objPos[2]> (self.objHeight + 0.005)) and grasped() :


                
                return hScale* min(heightTarget, objPos[2])
         
            else:
                return 0

        def placeReward(cond):

          
            c1 = 1000 ; c2 = 0.01 ; c3 = 0.001
           
            if cond:


                placeRew = 1000*(self.maxPlacingDist - placingDist) + c1*(np.exp(-(placingDist**2)/c2) + np.exp(-(placingDist**2)/c3))

               
                placeRew = max(placeRew,0)
           

                return placeRew

            else:
                return 0


        
        #print(self.maxPlacingDist)
        reachRew, reachDist = reachReward()

        if self.rewMode == 'orig':
            pickRew = orig_pickReward()
            placeRew  = placeReward(cond = self.pickCompleted and (graspDist < 0.1) and not(objDropped()))        

        else:

           
            assert(self.rewMode == 'general')
            pickRew = general_pickReward()
            placeRew  = placeReward(cond = self.pickCompleted and grasped())
       
     

        assert ((placeRew >=0) and (pickRew>=0))
        reward = reachRew + pickRew + placeRew

        #print(placingDist)
       

        return [reward, reachRew, reachDist, pickRew, placeRew, min(placingDist, self.maxPlacingDist)] 
     
