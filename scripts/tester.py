from multiworld.envs.mujoco.sawyer_xyz.sawyer_pick import SawyerPickEnv
import numpy as np
import time

env = SawyerPickEnv()
env.reset()

for i in range(int(1e6)):

    #actions = np.random.uniform(-0.5,0.5, size=(4))
    #env.step(actions)
   
    env.render()
    time.sleep(0.003)
    #time.sleep(0.045)
