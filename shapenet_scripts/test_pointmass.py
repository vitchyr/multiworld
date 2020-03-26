from roboverse.envs.pointmass import PointmassBaseEnv
import time
import numpy as np

env = PointmassBaseEnv(gui=True, observation_mode='state')
obs = env.reset()
action = np.zeros((2,))
for i in range(50):
    print(obs)
    obs, rew, done, info = env.step(action)
    goal_vec = env._goal_pos - obs
    action = 3*goal_vec[:2]
    env.render_obs()
    print(rew)
    print(info)
    time.sleep(0.5)
