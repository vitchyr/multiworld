import numpy as np
import pdb

import roboverse.bullet as bullet
import roboverse.bullet.objects as objects
import roboverse.devices as devices

space_mouse = devices.SpaceMouse()
space_mouse.start_control()

bullet.connect()
bullet.setup()

## load meshes
sawyer = objects.sawyer()
bowl = objects.bowl()
spam = objects.spam()
table = objects.table()

end_effector = bullet.get_index_by_attribute(sawyer, 'link_name', 'right_l6')
pos = np.array([0.5, 0, 0])
theta = [0.7071,0.7071,0,0]
bullet.position_control(sawyer, end_effector, pos, theta)


while True:

    delta = space_mouse.control
    z = delta[2]
    delta[2] = 0
    pos += delta * 0.1
    print(delta, pos)

    # bullet.sawyer_ik(sawyer, end_effector, pos, theta, space_mouse.control_gripper)
    bullet.sawyer_ik(sawyer, end_effector, pos, theta, z, gripper_bounds=[-1,1], discrete_gripper=False)
    bullet.step_ik()
    pos = bullet.get_link_state(sawyer, end_effector, 'pos')

    # bbox = bullet.get_bbox(cube, draw=True)
    # pdb.set_trace()
