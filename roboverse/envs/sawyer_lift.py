import numpy as np
import pdb

import roboverse.core as bullet
from roboverse.envs.sawyer_base import SawyerBaseEnv

class SawyerLiftEnv(SawyerBaseEnv):

    def __init__(self, goal_pos, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._goal_pos = goal_pos
        self._finger_tip = bullet.get_index_by_attribute(self._sawyer, 'link_name', 'right_gripper_r_finger_tip')

    def get_reward(self, observation):
        cube_pos = bullet.get_body_info(self._cube, 'pos')
        ee_pos = bullet.get_link_state(self._sawyer, self._finger_tip, 'pos')
        ee_dist = bullet.l2_dist(cube_pos, ee_pos)
        goal_dist = bullet.l2_dist(cube_pos, self._goal_pos)
        reward = -(ee_dist + goal_dist)
        return reward

if __name__ == "__main__":
    import time

    env = SawyerLiftEnv([.75,-.4,.2], render=False)
    # env.reset()

    ## interactive
    # import roboverse.devices as devices
    # space_mouse = devices.SpaceMouse()
    # space_mouse.start_control()

    # while True:
    #     delta = space_mouse.control
    #     gripper = space_mouse.control_gripper
    #     obs, rew, term, info = env.step(delta, gripper)
    #     img = env.render()
    #     # pdb.set_trace()
    #     print(rew, img.shape)

    ## simple timing
    num_steps = 100
    t0 = time.time()
    for i in range(num_steps):
        act = np.array([1.0, 0.0, 0.0, 0.0])
        obs, rew, term, info = env.step(act)
        print(i, obs.shape)
        img = env.render()
    t1 = time.time()

    tot_time = t1 - t0
    fps = num_steps / tot_time
    print('{} steps in {} seconds'.format(num_steps, tot_time))
    print('{} fps'.format(fps))