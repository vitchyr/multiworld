import roboverse
import time

env = roboverse.make('WidowBase-v0', gui=True)
env.reset()
for i in range(100000):
    env.step([0, ((i//10)%2) * 2 - 1, 0, (i//10)%2])
time.sleep(200000)

