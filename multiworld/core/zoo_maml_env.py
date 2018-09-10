
import numpy as np
from maml_zoo.logger import logger
from multiworld.core.wrapper_env import ProxyEnv


class ZooMamlEnv(ProxyEnv):

   
    def __init__(self, wrapped_env):
        self.quick_init(locals())
        super(ZooMamlEnv, self).__init__(wrapped_env)


    def set_task(self, task):
        """
        Args:
            task: task of the meta-learning environment

        """
        self.task = task
        self.change_task(task)
        

    def get_task(self):
        """
        Returns:
            task: task of the meta-learning environment
        """
        return self.task
   

    #@overrides
    def reset_model(self):

        #Task changing is done by set_task, and get_task

        self.reset_arm_and_object()
        return self._get_obs()


    def log_diagnostics(self, paths, prefix=''):

        self.wrapped_env.log_diagnostics(paths = paths, prefix = prefix, logger = logger)


        
    
    #required by rllab parallel sampler
    # def terminate(self):
    #     """
    #     Clean up operation,
    #     """
    #     pass
        
    

