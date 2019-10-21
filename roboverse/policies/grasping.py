import numpy as np
import roboverse.core as bullet

import pdb

class GraspingPolicy:

    def __init__(self, env, sawyer, obj):
        self._env = env
        self._sawyer = sawyer
        self._obj = obj

        self._env.open_gripper()
        self._gripper_width = self._env.get_gripper_width()

        self._end_effector = bullet.get_index_by_attribute(sawyer, 'link_name', 'right_gripper_l_finger_tip')
        self._ee_pos_fn = lambda: bullet.get_link_state(sawyer, self._end_effector, 'pos')
        # pdb.set_trace()
        self._obj_pos_fn = lambda: bullet.get_body_info(obj, 'pos')
        self._theta = bullet.deg_to_quat([180,0,180])
        self._mode = 0
        # pdb.set_trace()

    def control(self):
        ee_pos = np.array(self._ee_pos_fn())
        obj_pos = np.array(self._obj_pos_fn())

        l_finger_target = obj_pos.copy()
        l_finger_target[1] -= self._gripper_width / .4
        
        if self._mode == 0:
            l_finger_target[2] += 0.25
            gripper = -1
        elif self._mode == 1:
            gripper = -1
            l_finger_target[2] -= 0.03
            delta_pos = l_finger_target - ee_pos
            if np.abs(delta_pos)[2] < 0.015: self._mode += 1
        elif self._mode == 2:
            gripper = 1
            self._mode += 1
        elif self._mode == 3:
            l_finger_target[2] += 0.25
            l_finger_target[2] = np.clip(l_finger_target[2], 0, 0.25)
            gripper = 1


        delta_pos = l_finger_target - ee_pos
        if np.abs(delta_pos).max() < 9e-2: self._mode += 1

        if self._mode > 3:
            delta_pos[:] = 0
            gripper = 1
        # if self._mode > 0 or np.linalg.norm(delta_pos) < 1e-2:
            ## on top of object
            # delta_pos = obj_pos - ee_pos
            # delta_pos[1] -= 0.05
            # delta_pos[2] -= 0.25
            # delta_pos[2] = np.clip(delta_pos[2], -.05, .05)
            # delta_pos = np.array([0,0,-0.01])
            # self._mode = 1
        # else:
            # delta_pos = delta_pos / np.linalg.norm(delta_pos, 2) / 4.
            # pass

        print(self._mode, delta_pos)
        pos = ee_pos + delta_pos

        for _ in range(self._env._action_repeat):
            bullet.sawyer_ik(self._sawyer, self._end_effector, pos, self._theta, gripper, gripper_close_thresh=0)
            bullet.step_ik()

        # delta = np.array(obj_pos) - np.array(ee_pos)
        # pdb.set_trace()