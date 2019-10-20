import numpy as np
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

    p.setAdditionalSearchPath(pdata.getDataPath())
    # p.setAdditionalSearchPath('roboverse/envs/assets/')
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)


def setup(real_time=True, gravity=-10):
    p.setRealTimeSimulation(real_time)
    p.setGravity(0, 0, gravity)


def load_urdf(filepath, pos=[0, 0, 0], quat=[0, 0, 0, 1], scale=1, rgba=None):
    body = p.loadURDF(filepath, globalScaling=scale)
    p.resetBasePositionAndOrientation(body, pos, quat)
    if rgba is not None:
        p.changeVisualShape(body, -1, rgbaColor=rgba)
    return body


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

