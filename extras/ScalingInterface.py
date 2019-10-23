import os
from roboverse.core.ik import sawyer_ik, position_control
from roboverse.core.misc import load_urdf, load_obj, load_random_objects
from roboverse.core.queries import get_index_by_attribute, get_link_state
import numpy as np
import pybullet as p
import pybullet_data
from gym import spaces
from roboverse.envs.sawyer_reach import SawyerReachEnv
import roboverse.core
import pygame
from pygame.locals import QUIT, KEYDOWN, KEYUP

curr_dir = os.path.dirname(os.path.abspath(__file__))
home_dir = os.path.dirname(curr_dir)
filePath = home_dir + '/roboverse/envs/assets/ShapeNetCore'

objects = []
scaling = []
chosen_objects = []
for root, dirs, files in os.walk(filePath+'/ShapeNetCore_vhacd'):
    for d in dirs:
        for modelroot, modeldirs, modelfiles in os.walk(os.path.join(root, d)):
            for md in modeldirs:
                objects.append(os.path.join(modelroot, md))
                f = open(os.path.join(modelroot, md) + '/scale.txt')
                scaling.append(float(f.read()))
                f.close()
            break
    break

def newReset(self):
        p.resetSimulation()

        ## load meshes
        self._sawyer = load_urdf(self._sawyer_urdf_path)
        self._table = load_urdf(
            os.path.join(self._pybullet_data_dir, 'table/table.urdf'),
            [.75, -.2, -1], [0, 0, 0.707107, 0.707107],
            scale=1.0)
        self._end_effector = get_index_by_attribute(
            self._sawyer, 'link_name', 'right_l6')
       
        p.setPhysicsEngineParameter(numSolverIterations=150)
        p.setTimeStep(self._time_step)
        p.setGravity(0, 0, -10)
        p.stepSimulation()
        self.theta = [0.7071, 0.7071, 0, 0] 
        pos = np.array([0.5, 0, 0])
        position_control(self._sawyer, self._end_effector, pos, self.theta)
        return self.get_observation()

def renderObject(objectPath, scaling):
    path = objectPath.split('/')
    objectName = path[-1]
    dirName = path[-2]
    collisionPath = '{0}/ShapeNetCore_vhacd/{1}/{2}/model.obj'.format(filePath, dirName, objectName)
    visualPath =  '{0}/ShapeNetCore.v2/{1}/{2}/models/model_normalized.obj'.format(filePath, dirName, objectName)
    load_obj(collisionPath, visualPath, [.75, 0, 0.1], [0, 0, 0, 1], scale=scaling)
    p.stepSimulation()

SawyerReachEnv.reset = newReset
env = SawyerReachEnv(renders=True, control_xyz_position_only=False)
env.reset()
pygame.init()
screen = pygame.display.set_mode((100, 100))
time = pygame.time.get_ticks()
index = 0
scaleAmount = 0.1
renderObject(objects[index], scaling[index])

while True:
    for event in pygame.event.get():
        event_happened = True
        if event.type == QUIT:
            f = open(objects[index] + '/scale.txt', 'w')
            f.write(str(scaling[index]))   
            f.close()
            sys.exit()
        if event.type == KEYDOWN:
            if event.key == pygame.K_LEFT:
                f = open(objects[index] + '/scale.txt', 'w')
                f.write(str(scaling[index]))   
                f.close()
                index -= 1
                if index < 0:
                    index = len(objects) - 1
                env.reset()
                renderObject(objects[index], scaling[index])
            elif event.key == pygame.K_RIGHT:
                f = open(objects[index] + '/scale.txt', 'w')
                f.write(str(scaling[index]))   
                f.close()
                index += 1
                index %= len(objects)
                env.reset()
                renderObject(objects[index], scaling[index])
            elif event.key == pygame.K_UP:
                scaling[index] += scaleAmount
            elif event.key == pygame.K_DOWN:
                scaling[index] -= scaleAmount
            else:
                pressed = chr(event.dict['key'])
                if pressed == 's':
                    f = open(objects[index] + '/scale.txt', 'w')
                    f.write(str(scaling[index]))   
                    f.close()
                elif pressed == 'r':
                    env.reset()
                    renderObject(objects[index], scaling[index])
                elif pressed == 'o':
                    scaleAmount /= 2
                    print(scaleAmount)
                elif pressed == 'p':
                    scaleAmount *= 2
                    print(scaleAmount)
                              
    p.stepSimulation()
