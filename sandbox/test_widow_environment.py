import roboverse
import time

env = roboverse.make('WidowBase-v0', gui=True)
env.reset()
for i in range(100):
    env.step([0, 0, 0, 0])
time.sleep(200000)

