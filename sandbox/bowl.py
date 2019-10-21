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
# sawyer = bullet.load_urdf('roboverse/envs/assets/sawyer_robot/sawyer_description/urdf/sawyer_xacro.urdf')
sawyer = objects.sawyer()
# bowl = bullet.load_urdf('roboverse/envs/assets/objects/bowl.urdf', [.75, 0, -.3], scale=0.25)
bowl = objects.bowl()
# cube = bullet.load_urdf('roboverse/envs/assets/objects/cube.urdf', [.75, -.4, 0], scale=0.05)
spam = objects.spam()
# bowl = bullet.load_urdf('roboverse/envs/assets/objects/duck_prismatic.urdf', [.75, -.2, -.4], scale=0.5)
table = objects.table()
# table = bullet.load_urdf('table/table.urdf', [.75, -.2, -1], [0, 0, 0.707107, 0.707107], scale=1.0)
# duck = bullet.load_urdf('duck_vhacd.urdf', [.75, -.2, 0], [0, 0, 1, 0], scale=0.8)
# duck = bullet.load_urdf('lego/lego.urdf', [.75, .2, 0], [0, 0, 1, 0], rgba=[1,0,0,1], scale=1.5)

end_effector = bullet.get_index_by_attribute(sawyer, 'link_name', 'right_l6')
pos = np.array([0.5, 0, 0])
theta = [0.7071,0.7071,0,0]
bullet.position_control(sawyer, end_effector, pos, theta)


while True:

    delta = space_mouse.control
    pos += delta * 0.1
    print(delta, pos)

    bullet.sawyer_ik(sawyer, end_effector, pos, theta, space_mouse.control_gripper)
    bullet.step_ik()
    pos = bullet.get_link_state(sawyer, end_effector, 'pos')

    # bbox = bullet.get_bbox(cube, draw=True)
    # pdb.set_trace()
