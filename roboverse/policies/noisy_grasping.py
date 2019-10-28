import numpy as np
import roboverse.bullet as bullet

import pdb

class NoisyGraspingPolicy:

    def __init__(self, env, sawyer, obj):
        self._env = env
        self._sawyer = sawyer
        self._obj = obj
        self._goal_pos = np.array(env._goal_pos)
        self._gripper = -1
        self._l_finger = bullet.get_index_by_attribute(self._sawyer, 'link_name', 'right_gripper_l_finger')
        self._r_finger = bullet.get_index_by_attribute(self._sawyer, 'link_name', 'right_gripper_r_finger')
        self._ee_pos_fn = lambda: np.array(bullet.get_link_state(sawyer, env._end_effector, 'pos'))
        self._obj_pos_fn = lambda: np.array(bullet.get_midpoint(self._obj))

    def get_action(self, obs):
        obj_pos = self._obj_pos_fn()
        ee_pos = self._ee_pos_fn()

        delta_pos = obj_pos - ee_pos
        max_xy_delta = np.abs(delta_pos[:-1]).max()
        max_delta = np.abs(delta_pos).max()

        if max_xy_delta > .01:
            delta_pos[-1] = 0

        if (not self._gripper and max_delta < .01) or (self._gripper and max_delta < .04):
            self._gripper = 1
            delta_pos = np.clip((self._goal_pos - ee_pos), -1, 1)
        else:
            self._gripper = 0
            delta_pos = np.clip(delta_pos * 10, -1, 1)

        # print(delta_pos, self._gripper, max_delta)

        delta_pos += np.random.randn(3) * .1

        return delta_pos, self._gripper

