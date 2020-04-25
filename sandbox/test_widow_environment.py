import roboverse
import time
import pybullet as p

env = roboverse.make('WidowX200Grasp-v0', gui=True)
env.reset()

for i in range(1000):
    # env.step([0, ((i//10)%2) * 2 - 1, 0, (i//10)%2])
    env.step([0,0,0.1,0])
time.sleep(2)
