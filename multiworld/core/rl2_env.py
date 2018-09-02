from gym.spaces import  Dict

from gym.spaces import Box 


import numpy as np

from multiworld.core.wrapper_env import ProxyEnv


class Rl2Env(ProxyEnv):



    def __init__(self, wrapped_env):
        self.quick_init(locals())
        super(Rl2Env, self).__init__(wrapped_env)

    def reset_trial(self):

        self.sim.reset()


        task = self.sample_task()

        self.change_task(task)
        self.reset_arm_and_object()

        if self.viewer is not None:
            self.viewer_setup()
    
        return self.get_flat_obs()
      


    def reset(self):

        self.sim.reset()       
        self.reset_arm_and_object()

        if self.viewer is not None:
            self.viewer_setup()

        return self.get_flat_obs()



  