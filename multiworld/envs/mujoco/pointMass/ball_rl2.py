
from multiworld.envs.env_util import get_stat_in_paths, \
    create_stats_ordered_dict, get_asset_full_path

from multiworld.core.serializable import Serializable
from multiworld.envs.mujoco.mujoco_env import MujocoEnv

import numpy as np
import pickle

class BallEnv(MujocoEnv, Serializable):

   

    def __init__(self, init_pos = [0,0] ,num_goals = 20, *args, **kwargs):
        
        
      
        model_name = get_asset_full_path('pointMass/ball.xml')
        self.obj_init_pos = init_pos
        

        self.goals = pickle.load(open("/home/russellm/multiworld/multiworld/envs/goals/PointMassGoals.pkl", "rb"))

        self.num_goals =min(num_goals, len(self.goals))

        self.goalPos = self.sample_goal()
       

        self.curr_path_length = 0
        self.max_path_length = 100

        MujocoEnv.__init__(self, model_name, frame_skip=1, automatically_set_spaces=True)
        Serializable.__init__(self, *args, **kwargs)

       
        self.reset_trial()
        
        
      

    
    def _get_obs(self):
       

        return np.concatenate([
            self.data.qpos.flat,
            self.data.qvel.flat,
        ])

    def viewer_setup(self):
        
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = 0.8
        self.viewer.cam.azimuth = 90.0
        self.viewer.cam.elevation = -90.0
        self.viewer.cam.lookat[0] = 0
        self.viewer.cam.lookat[1] = 0
        self.viewer.cam.lookat[2] = 0
    

    def get_site_pos(self, siteName):

       
        _id = self.model.site_names.index(siteName)
        return self.data.site_xpos[_id].copy()



    def step(self, action):
        
       
      
        self.do_simulation(action)
        
        ballPos = self.get_body_com("ball")
        goalPos = self.goalPos
        obs = self._get_obs()

        

        reward = -np.linalg.norm(ballPos - goalPos)
       
        
        self.curr_path_length +=1

        if self.curr_path_length == self.max_path_length:
            done = True
        else:
            done = False

        return obs, reward, done, {}

    def _set_obj(self, pos):
        
        
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()

        qpos[0:2] = pos.copy()
        qvel[0:2] = 0
        self.set_state(qpos, qvel)

    def _set_goal_marker(self, goal):
        """
        This should be use ONLY for visualization. Use self._state_goal for
        logging, learning, etc.
        """

       
        self.model.site_pos[self.model.site_name2id('goal')] = (
            goal[:3]
        )


    def sample_goal(self):

        goal_idx = np.random.randint(0, self.num_goals)
        goal = self.goals[goal_idx]
        return [goal[0], goal[1], 0]



    def reset_trial(self):
        self.sim.reset()
        self.goalPos = self.sample_goal()


        self._set_obj(self.obj_init_pos)
        self._set_goal_marker(self.goalPos)

       

        self.curr_path_length = 0

        if self.viewer is not None:
            self.viewer_setup()

        return self._get_obs()


    def reset(self):
        self.sim.reset()
      
        self._set_obj(self.obj_init_pos)
        self._set_goal_marker(self.goalPos)

       

        self.curr_path_length = 0

        if self.viewer is not None:
            self.viewer_setup()

        return self._get_obs()

