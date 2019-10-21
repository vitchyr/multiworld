import time
import numpy as np
import pdb

import roboverse.core as bullet
import roboverse.core.objects as objects
import roboverse.devices as devices

space_mouse = devices.SpaceMouse()
space_mouse.start_control()

bullet.connect()
bullet.setup()

## load meshes
table = objects.table()
spam = objects.spam()

while True:
    time.sleep(0.01)
    bullet.step()
