
from multiworld.envs.env_util import get_stat_in_paths, \
    create_stats_ordered_dict, get_asset_full_path

from multiworld.core.serializable import Serializable
from multiworld.envs.mujoco.mujoco_env import MujocoEnv

import numpy as np

class BallEnv(MujocoEnv, Serializable):

   

    def __init__(self, init_pos = [0,0] , goal_pos = [0.2, 0], *args, **kwargs):
        
        
      
        model_name = get_asset_full_path('pointMass/ball.xml')
        self.obj_init_pos = init_pos
        self.goalPos = [goal_pos[0] , goal_pos[1], 0]

        self.curr_path_length = 0
        self.max_path_length = 120

        MujocoEnv.__init__(self, model_name, frame_skip=1, automatically_set_spaces=True)
        Serializable.__init__(self, *args, **kwargs)


        self.reset()
        
        
        # self.get_viewer()
        # self.viewer_setup()

    
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
    
    def reset_model(self):

      
        self._set_obj(self.obj_init_pos)
        self._set_goal_marker(self.goalPos)

       

        self.curr_path_length = 0
        return self._get_obs()


