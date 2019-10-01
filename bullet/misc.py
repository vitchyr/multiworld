import numpy as np
import pdb

import pybullet as p
import pybullet_data as pdata

#########################
#### setup functions ####
#########################

def connect():
    clid = p.connect(p.SHARED_MEMORY)
    if (clid<0):
        p.connect(p.GUI)

    p.setAdditionalSearchPath(pdata.getDataPath())
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0)
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1)

def setup(real_time=True, gravity=-10):
    p.setRealTimeSimulation(real_time)
    p.setGravity(0,0,gravity)

def load_urdf(filepath, pos=[0,0,0], quat=[0,0,0,1], scale=1, rgba=None):
    body = p.loadURDF(filepath, globalScaling=scale)
    p.resetBasePositionAndOrientation(body, pos, quat)
    if rgba is not None:
        p.changeVisualShape(body, -1, rgbaColor=rgba)
    return body

############################
#### rotation functions ####
############################

def deg_to_rad(deg):
    return [d * np.pi / 180. for d in deg]

def rad_to_deg(rad):
    return [r * 180. / np.pi for r in rad]

def quat_to_deg(quat):
    euler_rad = p.getEulerFromQuaternion(quat)
    euler_deg = rad_to_deg(euler_rad)
    return euler_deg

def deg_to_quat(deg):
    rad = deg_to_rad(deg)
    quat = p.getQuaternionFromEuler(rad)
    return quat

#######################
#### miscellaneous ####
#######################

def step():
    p.stepSimulation()



    
# space_mouse = SpaceMouse()
# space_mouse.start_control()

# connect()
# setup()
# sawyer = load_urdf('sawyer_robot/sawyer_description/urdf/sawyer_xacro.urdf')
# table = load_urdf('table/table.urdf', [.75, -.2, -1], [0, 0, 0.707107, 0.707107], scale=1.0)
# duck = load_urdf('duck_vhacd.urdf', [.75, -.2, 0], [0, 0, 1, 0], scale=0.8)

# end_effector = get_index_by_attribute(sawyer, 'link_name', 'right_l6')
# pos = np.array([0.5, 0, 0])
# theta = [0.7071,0.7071,0,0]
# position_control(sawyer, end_effector, pos, theta)


# while True:

#     delta = space_mouse.control
#     pos += delta * 0.1
#     print(delta, pos)

#     sawyer_ik(sawyer, end_effector, pos, theta, space_mouse.control_gripper)
#     p.stepSimulation()
#     pos = get_link_state(sawyer, end_effector, 'pos')


#     # #### blender
#     # sawyerId = 0
#     # visual_data = p.getVisualShapeData(sawyerId)
#     # visual_data = {k[1]: (k[-3], quat_to_degrees(k[-2])) for k in visual_data}

#     # num_joints = p.getNumJoints(sawyerId)
#     # for i in range(num_joints):
#     #     if i not in visual_data:
#     #         visual_data[i] = ([0,0,0], [0,0,0])

#     # link_names = [p.getJointInfo(0, i)[12].decode() for i in range(num_joints)]
#     # link_d = {link: get_link_state(0, i, visual_data[i]) for i, link in enumerate(link_names)}
    
    # pdb.set_trace()
# # state = 0

# increasing = True
# while True:
#     p.stepSimulation()
#     time.sleep(0.01)

#     if state < high and increasing:
#         state = np.clip(state+0.001, low, high)
#         if state == high: increasing = False

#     if state > low and not increasing:
#         state = np.clip(state-0.001, low, high)
#         if state == 0: increasing = True

#     p.resetJointState(0, joint_ind, state)
#     p.resetJointState(0, joint_ind+2, -state)   

#     print(state, low, high)
 
# clid = p.connect(p.SHARED_MEMORY)

# if (clid<0):
#     p.connect(p.GUI)

# p.setAdditionalSearchPath(pdata.getDataPath())
# # p.loadURDF("plane.urdf",[0,0,-.98])
# p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0)
# # sawyerId = p.loadURDF("sawyer_robot/sawyer_description/urdf/sawyer.urdf",[0,0,0])
# sawyerId = p.loadURDF('sawyer_robot/sawyer_description/urdf/sawyer_xacro.urdf')
# p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1)
# p.resetBasePositionAndOrientation(sawyerId,[0,0,0],[0,0,0,1])

# pdb.set_trace()

# pdb.set_trace()
#bad, get it from name! sawyerEndEffectorIndex = 18
# sawyerEndEffectorIndex = 16
# num_joints = p.getNumJoints(0)
# sawyerEndEffectorIndex = p.getJointInfo(0, num_joints-1)
# sawyerEndEffectorIndex = 16
# numJoints = p.getNumJoints(sawyerId)
#joint damping coefficents
# jd=[0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001]

# p.setGravity(0,0,-10)
# t=0.
# prevPose=[0,0,0]
# prevPose1=[0,0,0]
# hasPrevPose = 0

# # useRealTimeSimulation = 1
# # p.setRealTimeSimulation(useRealTimeSimulation)
# #trailDuration is duration (in seconds) after debug lines will be removed automatically
# #use 0 for no-removal
# trailDuration = 15
    
# space_mouse = SpaceMouse()
# space_mouse.start_control()

# # sphere_collision = p.createCollisionShape(p.GEOM_SPHERE, radius=0.05)
# # sphere_visual = p.createVisualShape(p.GEOM_SPHERE, radius=0.05, rgbaColor=[1,1,0,1])
# # mass = 0
# # sphere = p.createMultiBody(mass, sphere_collision, sphere_visual)
# # _, sphere_orientation = p.getBasePositionAndOrientation(sphere)

# # objects = [
# p.loadURDF("table/table.urdf", [0.7500000, -0.200000, -1.000000], [0.000000, 0.000000, 0.707107, 0.707107], globalScaling=1.0)
# p.loadURDF("duck_vhacd.urdf", [0.7500000, -0.200000, 0.0000000], [0.000000, 0.000000, 1, 0], globalScaling=0.8)

# pdb.set_trace()


# # pos = np.array([0.5, 0, 0])
# # ik_solution = p.calculateInverseKinematics(sawyerId,sawyerEndEffectorIndex,pos,targetOrientation=[0.7071,0.7071,0,0],jointDamping=jd)
# # indices, velocities = get_joint_velocities(ik_solution)
# # for ind, jpos in zip(indices, ik_solution):
# #     p.resetJointState(0, ind, jpos)

# # pos = np.array([0.5, 0, 0])
# # position_control(sawyer, end_effector, pos)

# # mesh = p.getMeshData(1)

# while 1:
#     if (useRealTimeSimulation):
#         dt = datetime.now()
#         t = (dt.second/60.)*2.*math.pi
#         print (t)
#     else:
#         t=t+0.01
#         time.sleep(0.01)
    
#     for i in range (1):
#         # pos = [1.0,0.2*math.cos(t),0.+0.2*math.sin(t)]
#         # pos = [0.75,0.2*math.cos(t),0.+0.2*math.sin(t)]
#         delta = space_mouse.control
#         pos += delta * 0.1
#         # p.resetBasePositionAndOrientation(sphere, pos, sphere_orientation)
#         print(delta, pos)
#         ik_solution = p.calculateInverseKinematics(sawyerId,sawyerEndEffectorIndex,pos,targetOrientation=[0.7071,0.7071,0,0],jointDamping=jd)

#         gripper_state = set_gripper(space_mouse.control_gripper)
#         ik_solution = list(ik_solution)
#         ik_solution[-2:] = gripper_state

#         indices, velocities = get_joint_velocities(ik_solution)

#         # pdb.set_trace()

#         p.setJointMotorControlArray(0, indices, p.VELOCITY_CONTROL, targetVelocities=velocities)
#         p.stepSimulation()

#         # for ind, vel in zip(indices, velocities):
#             # p.resetJointState(0, ind, ik_solution)

#         # for ind, jpos in zip(indices, ik_solution):
#         #   p.resetJointState(0, ind, jpos)

#         # pdb.set_trace()
#         #reset the joint state (ignoring all dynamics, not recommended to use during simulation)
#         # jointPoses = ik_solution
#         # for i in range (numJoints):
#         #   jointInfo = p.getJointInfo(sawyerId, i)
#         #   qIndex = jointInfo[3]
#         #   if qIndex > -1:
#         #       print(jointPoses)
#         #       print(qIndex)
#         #       print(qIndex-7)
#         #       p.resetJointState(sawyerId,i,jointPoses[qIndex-7])
                
#                 # current = get_sawyer_jpos()
#         # pdb.set_trace()

#         # set_gripper(space_mouse.control_gripper)

#     ls = p.getLinkState(sawyerId,sawyerEndEffectorIndex)
#     if (hasPrevPose):
#         p.addUserDebugLine(prevPose,pos,[0,0,0.3],1,trailDuration)
#         p.addUserDebugLine(prevPose1,ls[4],[1,0,0],1,trailDuration)
#     prevPose=pos
#     prevPose1=ls[4]
#     hasPrevPose = 1 

#     # pdb.set_trace()   

#     # pos = p.getLinkState(0,24)[4]
#     # print(pos)
#     pos = ls[0]

#     #### blender
#     sawyerId = 0
#     visual_data = p.getVisualShapeData(sawyerId)
#     visual_data = {k[1]: (k[-3], quat_to_degrees(k[-2])) for k in visual_data}

#     num_joints = p.getNumJoints(sawyerId)
#     for i in range(num_joints):
#         if i not in visual_data:
#             visual_data[i] = ([0,0,0], [0,0,0])

#     link_names = [p.getJointInfo(0, i)[12].decode() for i in range(num_joints)]
#     link_d = {link: get_link_state(0, i, visual_data[i]) for i, link in enumerate(link_names)}
    
#     pdb.set_trace()




