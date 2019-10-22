import numpy as np
import random
import os
import pdb

import pybullet as p
import pybullet_data as pdata


#########################
#### setup functions ####
#########################

def connect():
    clid = p.connect(p.SHARED_MEMORY)
    if (clid < 0):
        p.connect(p.GUI)

    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
    p.resetDebugVisualizerCamera(0.8, 90, -20, [0.75, -.2, 0])
    p.setAdditionalSearchPath(pdata.getDataPath())
    # p.setAdditionalSearchPath('roboverse/envs/assets/')

def connect_headless(render=False):
    if render:
        cid = p.connect(p.SHARED_MEMORY)
        if cid < 0:
            p.connect(p.GUI)
    else:
        p.connect(p.DIRECT)

    p.resetDebugVisualizerCamera(0.8, 90, -20, [0.75, -.2, 0])
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
    body = p.loadURDF(filepath, globalScaling=scale)
    p.resetBasePositionAndOrientation(body, pos, quat)
    if rgba is not None:
        p.changeVisualShape(body, -1, rgbaColor=rgba)
    return body

def load_obj(filepathcollision, filepathvisual, pos=[0, 0, 0], quat=[0, 0, 0, 1], scale=1, rgba=None):
    collisionid= p.createCollisionShape(p.GEOM_MESH, fileName=filepathcollision, meshScale=scale * np.array([1, 1, 1]))
    visualid = p.createVisualShape(p.GEOM_MESH, fileName=filepathvisual, meshScale=scale * np.array([1, 1, 1]))
    body = p.createMultiBody(0.05, collisionid, visualid)
    p.resetBasePositionAndOrientation(body, pos, quat)
    return body


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
           shadow=1, light_direction=[1,1,1], renderer=p.ER_BULLET_HARDWARE_OPENGL):
    ## ER_BULLET_HARDWARE_OPENGL
    img_tuple = p.getCameraImage(width,
                                 height,
                                 view_matrix,
                                 projection_matrix,
                                 shadow=shadow,
                                 lightDirection=light_direction,
                                 renderer=renderer)
    _, _, img, depth, segmentation = img_tuple
    img = img[:,:,:-1]
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

def get_bbox(body, draw=False):
    xyz_min, xyz_max = p.getAABB(body)
    if draw:
        draw_bbox(xyz_min, xyz_max)
    return np.array(xyz_min), np.array(xyz_max)

def get_midpoint(body):
    xyz_min, xyz_max = get_bbox(body)
    midpoint = (xyz_min + xyz_max) / 2.
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

def load_random_objects(filePath, number):
    objects = []
    chosen_objects = []
    print(filePath)
    for root, dirs, files in os.walk(filePath+'/ShapeNetCore.v2'):
        for d in dirs:
            for modelroot, modeldirs, modelfiles in os.walk(os.path.join(root, d)):
                for md in modeldirs:
                    objects.append(os.path.join(modelroot, md))
                break
        break
    try:
        chosen_objects = random.sample(range(len(objects)), number) 
    except ValueError:
        print('Sample size exceeded population size')
    
    object_ids = []
    count = 1
    for i in chosen_objects:
        path = objects[i].split('/')
        dirName = path[11]
        objectName = path[12]
        f = open(filePath+'/ShapeNetCore_vhacd/{0}/{1}/scale.txt'.format(dirName, objectName), 'r')
        scaling = float(f.read()) 
        obj = load_obj(filePath+'/ShapeNetCore_vhacd/{0}/{1}/model.obj'.format(dirName, objectName),
            filePath+'/ShapeNetCore.v2/{0}/{1}/models/model_normalized.obj'.format(dirName, objectName),
            [random.uniform(0.65, 0.85), random.uniform(-0.5, 0.5), 0], [0, 0, 1, 0], scale=scaling)
        object_ids.append(obj)
        count += 1
    return object_ids
