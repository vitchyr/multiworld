import os
import random
import numpy as np
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in os.sys.path:
    os.sys.path.remove(ros_path)
import cv2
os.sys.path.append(ros_path)
import pdb

import pybullet as p
import pybullet_data as pdata

from roboverse.utils.serialization import make_dir


#########################
#### setup functions ####
#########################

def connect():
    clid = p.connect(p.SHARED_MEMORY)
    if (clid < 0):
        p.connect(p.GUI)

    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

    p.resetDebugVisualizerCamera(cameraDistance=0.3, cameraYaw=90, cameraPitch=-15, cameraTargetPosition=[.7, 0, -0.3])
    #p.resetDebugVisualizerCamera(0.8, 90, -20, [0.75, -.2, 0])
    p.setAdditionalSearchPath(pdata.getDataPath())

def connect_headless(gui=False):
    if gui:
        cid = p.connect(p.SHARED_MEMORY)
        if cid < 0:
            p.connect(p.GUI)
    else:
        p.connect(p.DIRECT)

    p.resetDebugVisualizerCamera(cameraDistance=0.3, cameraYaw=90, cameraPitch=-15, cameraTargetPosition=[.7, 0, -0.3])
    #p.resetDebugVisualizerCamera(0.8, 90, -20, [0.75, -.2, 0])
    p.setAdditionalSearchPath(pdata.getDataPath())


def setup(real_time=True, gravity=-10):
    '''
        sets parameters for running pybullet 
        interactively
    '''
    p.setRealTimeSimulation(real_time)
    p.setGravity(0, 0, gravity)
    p.stepSimulation()

def setup_headless(timestep=1./240, solver_iterations=150, gravity=-10):
    '''
        sets parameters for running pybullet 
        in a headless environment
    '''
    p.setPhysicsEngineParameter(numSolverIterations=solver_iterations)
    p.setTimeStep(timestep)
    p.setGravity(0, 0, gravity)
    p.stepSimulation()

def reset():
    p.resetSimulation()

def load_urdf(filepath, pos=[0, 0, 0], quat=[0, 0, 0, 1], scale=1, rgba=None):
    #rgba = list(np.random.choice(range(256), size=3) / 255.0) + [1]

    body = p.loadURDF(filepath, globalScaling=scale)
    p.resetBasePositionAndOrientation(body, pos, quat)
    if rgba is not None:
        p.changeVisualShape(body, -1, rgbaColor=rgba)

    return body

def load_obj(filepathcollision, filepathvisual, pos=[0, 0, 0], quat=[0, 0, 0, 1], scale=1, rgba=None):
    collisionid= p.createCollisionShape(p.GEOM_MESH, fileName=filepathcollision, meshScale=scale * np.array([1, 1, 1]))
    visualid = p.createVisualShape(p.GEOM_MESH, fileName=filepathvisual, meshScale=scale * np.array([1, 1, 1]))
    body = p.createMultiBody(0.05, collisionid, visualid)
    if rgba is not None:
        p.changeVisualShape(body, -1, rgbaColor=rgba)
    p.resetBasePositionAndOrientation(body, pos, quat)
    return body

def save_state(*savepath):
    if len(savepath) > 0:
        savepath = os.path.join(*savepath)
        make_dir(os.path.dirname(savepath))
        p.saveBullet(savepath)
        state_id = None
    else:
        state_id = p.saveState()
    return state_id

def load_state(*loadpath):
    loadpath = os.path.join(*loadpath)
    p.restoreState(fileName=loadpath)

#############################
#### rendering functions ####
#############################

def get_view_matrix(target_pos=[.75, -.2, 0], distance=0.9, 
                    yaw=90, pitch=-20, roll=0, up_axis_index=2):
    view_matrix = p.computeViewMatrixFromYawPitchRoll(
        target_pos, distance, yaw, pitch, roll, up_axis_index)
    return view_matrix

def get_projection_matrix(height, width, fov=60, near_plane=0.1, far_plane=2):
    aspect = width / height
    projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near_plane, far_plane)
    return projection_matrix

def render(height, width, view_matrix, projection_matrix, 
           shadow=1, light_direction=[1,1,1], renderer=p.ER_BULLET_HARDWARE_OPENGL, gaussian_width=5):
    ## ER_BULLET_HARDWARE_OPENGL
    img_tuple = p.getCameraImage(width,
                                 height,
                                 view_matrix,
                                 projection_matrix,
                                 shadow=shadow,
                                 lightDirection=light_direction,
                                 renderer=renderer)
    _, _, img, depth, segmentation = img_tuple
    # import ipdb; ipdb.set_trace()
    # Here, if I do len(img), I get 9216.
    # img = np.reshape(np.array(img), (48, 48, 4))
    img = img[:,:,:-1]
    if gaussian_width > 0:
        img = cv2.GaussianBlur(img, (gaussian_width, gaussian_width), 0)
    return img, depth, segmentation

############################
#### rotation functions ####
############################

def deg_to_rad(deg):
    return np.array([d * np.pi / 180. for d in deg])


def rad_to_deg(rad):
    return np.array([r * 180. / np.pi for r in rad])


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

def l2_dist(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.linalg.norm(a-b, 2)

def rot_diff_deg(a, b):
    '''
        a, b : orientations in degrees
        returns ||a-b||_1, taking into account that multiple
        [r_x, r_y, r_z] vectors correspond to the same orientation
    '''
    diff = (a - b) % 360
    diff = np.minimum(diff, 360-diff)
    return np.linalg.norm(diff, 1)

def add_debug_line(x, y, rgb=[1,0,0], duration=5):
    p.addUserDebugLine(x, y, rgb, duration)

# def is_contacting(body_1, body_2, link_1=-1, link_2=-1):
#     points = p.getContactPoints(body_1, body_2, link_1, link_2)
#     return len(points) > 0

def is_contacting(body_1, body_2, link_1=-1, link_2=-1, threshold=.005):
    dist = get_link_dist(body_1, body_2, link_1=link_1, link_2=link_2)
    return dist < threshold

def get_link_dist(body_1, body_2, link_1=-1, link_2=-1, threshold=1):
    points = p.getClosestPoints(body_1, body_2, threshold, link_1, link_2)
    distances = [point[8] for point in points] + [np.float('inf')]
    return min(distances)

def get_bbox(body, draw=False):
    xyz_min, xyz_max = p.getAABB(body)
    if draw:
        draw_bbox(xyz_min, xyz_max)
    return np.array(xyz_min), np.array(xyz_max)

def bbox_intersecting(bbox_1, bbox_2):
    min_1, max_1 = bbox_1
    min_2, max_2 = bbox_2
    # print(min_1, max_1, min_2, max_2)
    intersecting = (min_1 <= max_2).all() and (min_2 <= max_1).all()
    return intersecting

def get_midpoint(body, weights=[.5,.5,.5]):
    weights = np.array(weights)
    xyz_min, xyz_max = get_bbox(body)
    midpoint = xyz_max * weights + xyz_min * (1 - weights)
    return midpoint

def draw_bbox(aabbMin, aabbMax):
    '''
        https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/examples/getAABB.py
    '''
    f = [aabbMin[0], aabbMin[1], aabbMin[2]]
    t = [aabbMax[0], aabbMin[1], aabbMin[2]]
    p.addUserDebugLine(f, t, [1, 0, 0])
    f = [aabbMin[0], aabbMin[1], aabbMin[2]]
    t = [aabbMin[0], aabbMax[1], aabbMin[2]]
    p.addUserDebugLine(f, t, [0, 1, 0])
    f = [aabbMin[0], aabbMin[1], aabbMin[2]]
    t = [aabbMin[0], aabbMin[1], aabbMax[2]]
    p.addUserDebugLine(f, t, [0, 0, 1])

    f = [aabbMin[0], aabbMin[1], aabbMax[2]]
    t = [aabbMin[0], aabbMax[1], aabbMax[2]]
    p.addUserDebugLine(f, t, [1, 1, 1])

    f = [aabbMin[0], aabbMin[1], aabbMax[2]]
    t = [aabbMax[0], aabbMin[1], aabbMax[2]]
    p.addUserDebugLine(f, t, [1, 1, 1])

    f = [aabbMax[0], aabbMin[1], aabbMin[2]]
    t = [aabbMax[0], aabbMin[1], aabbMax[2]]
    p.addUserDebugLine(f, t, [1, 1, 1])

    f = [aabbMax[0], aabbMin[1], aabbMin[2]]
    t = [aabbMax[0], aabbMax[1], aabbMin[2]]
    p.addUserDebugLine(f, t, [1, 1, 1])

    f = [aabbMax[0], aabbMax[1], aabbMin[2]]
    t = [aabbMin[0], aabbMax[1], aabbMin[2]]
    p.addUserDebugLine(f, t, [1, 1, 1])

    f = [aabbMin[0], aabbMax[1], aabbMin[2]]
    t = [aabbMin[0], aabbMax[1], aabbMax[2]]
    p.addUserDebugLine(f, t, [1, 1, 1])

    f = [aabbMax[0], aabbMax[1], aabbMax[2]]
    t = [aabbMin[0], aabbMax[1], aabbMax[2]]
    p.addUserDebugLine(f, t, [1.0, 0.5, 0.5])
    f = [aabbMax[0], aabbMax[1], aabbMax[2]]
    t = [aabbMax[0], aabbMin[1], aabbMax[2]]
    p.addUserDebugLine(f, t, [1, 1, 1])
    f = [aabbMax[0], aabbMax[1], aabbMax[2]]
    t = [aabbMax[0], aabbMax[1], aabbMin[2]]
    p.addUserDebugLine(f, t, [1, 1, 1])
