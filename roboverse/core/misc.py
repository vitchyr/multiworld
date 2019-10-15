import numpy as np

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
