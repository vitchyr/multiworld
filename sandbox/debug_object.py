import time
import numpy as np
import pdb

import roboverse.core as bullet
import roboverse.devices as devices

space_mouse = devices.SpaceMouse()
space_mouse.start_control()

bullet.connect()
bullet.setup()

## load meshes
# sawyer = bullet.load_urdf('roboverse/envs/assets/sawyer_robot/sawyer_description/urdf/sawyer_xacro.urdf')
# bowl = bullet.load_urdf('roboverse/envs/assets/objects/bowl.urdf', [.75, 0, -.3], scale=0.25)
cube = bullet.load_urdf('roboverse/envs/assets/objects/spam/spam.urdf', [.75, -.4, 0], scale=0.05)
# bowl = bullet.load_urdf('roboverse/envs/assets/objects/duck_prismatic.urdf', [.75, -.2, -.6], scale=1.0)
table = bullet.load_urdf('table/table.urdf', [.75, -.2, -1], [0, 0, 0.707107, 0.707107], scale=1.0)
# duck = bullet.load_urdf('duck_vhacd.urdf', [.75, -.2, 0], [0, 0, 1, 0], scale=0.8)
# duck = bullet.load_urdf('lego/lego.urdf', [.75, .2, 0], [0, 0, 1, 0], rgba=[1,0,0,1], scale=1.5)


while True:
    time.sleep(0.01)
    bullet.step()
