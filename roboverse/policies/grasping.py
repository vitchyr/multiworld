import numpy as np
import roboverse.bullet as bullet

import pdb

class GraspingPolicy:

    def __init__(self, env, sawyer, obj, sigma=0.1, 
                 gripper_close_delta=0.01, gripper_remain_closed_delta=0.04, 
                 verbose=False):
        self._env = env
        self._sawyer = sawyer
        self._obj = obj
        self._gripper_open, self._gripper_close = self._env._gripper_bounds
        self._sigma = sigma
        self._gripper_close_delta = gripper_close_delta
        self._gripper_remain_closed_delta = gripper_remain_closed_delta
        self._verbose = verbose
        self._goal_pos = np.array(env._goal_pos)
        self._gripper = self._gripper_open
        self._l_finger = bullet.get_index_by_attribute(self._sawyer, 'link_name', 'right_gripper_l_finger')
        self._r_finger = bullet.get_index_by_attribute(self._sawyer, 'link_name', 'right_gripper_r_finger')
        self._ee_pos_fn = lambda: np.array(bullet.get_link_state(sawyer, env._end_effector, 'pos'))
        self._obj_pos_fn = lambda: np.array(bullet.get_midpoint(self._obj, weights=[.5,.5,.5]))

    def get_action(self, obs):
        obj_pos = self._obj_pos_fn()
        ee_pos = self._ee_pos_fn()

        delta_pos = obj_pos - ee_pos
        max_xy_delta = np.abs(delta_pos[:-1]).max()
        max_delta = np.abs(delta_pos).max()

        if max_xy_delta > .01:
            delta_pos[-1] = 0

        if (self._gripper == self._gripper_open and max_delta < self._gripper_close_delta) or \
           (self._gripper == self._gripper_close and max_delta < self._gripper_remain_closed_delta):
            self._gripper = self._gripper_close
            delta_pos = np.clip((self._goal_pos - ee_pos), -1, 1)
        else:
            self._gripper = self._gripper_open
            delta_pos = np.clip(delta_pos * 10, -1, 1)

        if self._verbose:
            print(delta_pos, self._gripper, max_delta)

        noise = np.random.randn(3) * self._sigma
        delta_pos += noise  

        action = np.concatenate((delta_pos, np.array([self._gripper])))
        #pdb.set_trace()
        # return delta_pos, self._gripper
        return action

