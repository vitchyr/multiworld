import time
import numpy as np
import pdb

import roboverse.bullet as bullet
import roboverse.devices as devices

# space_mouse = devices.SpaceMouse()
# space_mouse.start_control()

bullet.connect()
bullet.setup()

## load meshes
# sawyer = bullet.objects.sawyer(quat=[0,0,1,0])
# table = bullet.objects.table()
# spam = bullet.objects.spam()

# bullet.load_obj()

import pybullet as p
for i in range(64):
	filename = 'dump/counters/{}.obj'.format(i)
	scale = 0.075
	collision_shape = p.createCollisionShape(p.GEOM_MESH, fileName=filename, meshScale=[scale,scale,scale])
	visual_shape = p.createVisualShape(p.GEOM_MESH, fileName=filename, meshScale=[scale,scale,scale])
	mass = 0
	p.createMultiBody(mass, collision_shape, visual_shape, baseOrientation=[.707,0,0,.707], basePosition=[-.75,0,-.75])

while True:
    time.sleep(0.01)
    bullet.step()
    # pdb.set_trace()
