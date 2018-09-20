from gym.spaces import  Dict
from gym.spaces import Box 
import numpy as np
from multiworld.core.wrapper_env import ProxyEnv
import rllab.misc.logger as logger


class FinnMamlEnv(ProxyEnv):

   
    def __init__(self, wrapped_env):
        self.quick_init(locals())
        self._goal_idx = 3
        super(FinnMamlEnv, self).__init__(wrapped_env)


    def sample_goals(self, num_goals):

        return np.array(range(num_goals))
       
    #@overrides
    def reset(self, reset_args = None):
        self.sim.reset()

        goal_idx = reset_args
        if goal_idx is not None:
            self._goal_idx = goal_idx
        elif self._goal_idx is None:
            self._goal_idx = np.random.randint(1)
   
        task = self.tasks[self._goal_idx]
        self.change_task(task)
        self.reset_arm_and_object()

        if self.viewer is not None:
            self.viewer_setup()

        return self.get_flat_obs()
   
  
    def log_diagnostics(self, paths, prefix=''):

        self.wrapped_env.log_diagnostics(paths = paths, prefix = prefix, logger = logger)
    
    #required by rllab parallel sampler
    def terminate(self):
        """
        Clean up operation,
        """
        pass
        
    

