import os
import sys
from roboverse.bullet.ik import sawyer_ik, position_control
from roboverse.bullet.misc import load_urdf, load_obj
from roboverse.bullet.queries import get_index_by_attribute, get_link_state
import numpy as np
import pybullet as p
from roboverse.envs.sawyer_reach import SawyerReachEnv
import pygame
from pygame.locals import QUIT, KEYDOWN, KEYUP
import json

curr_dir = os.path.dirname(os.path.abspath(__file__))
home_dir = os.path.dirname(curr_dir)
filePath = home_dir + '/roboverse/envs/assets/ShapeNetCore'
jsonPath = filePath + '/scaling.json'

objects = []
with open(jsonPath, 'r') as fp:
    scaling = json.load(fp)
    
chosen_objects = []
for root, dirs, files in os.walk(filePath+'/ShapeNetCore_vhacd'):
    for d in dirs:
        for modelroot, modeldirs, modelfiles in os.walk(os.path.join(root, d)):
            for md in modeldirs:
                objects.append(os.path.join(modelroot, md))
            break
    break

def new_reset(self):
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

def render_object(objectPath, scaling):
    path = objectPath.split('/')
    objectName = path[-1]
    dirName = path[-2]
    collisionPath = '{0}/ShapeNetCore_vhacd/{1}/{2}/model.obj'.format(filePath, dirName, objectName)
    visualPath =  '{0}/ShapeNetCore.v2/{1}/{2}/models/model_normalized.obj'.format(filePath, dirName, objectName)
    load_obj(collisionPath, visualPath, [.75, 0, 0.15], [0, 0, 0, 1], scale=scaling)
    p.stepSimulation()

def json_key(i):
    objectName = os.path.basename(objects[i])
    dirName = os.path.basename(os.path.dirname(objects[i]))
    return '{0}/{1}'.format(dirName, objectName)
    
def json_dump():
    global jsonPath, scaling 
    with open(jsonPath, 'w') as fp:
        json.dump(scaling, fp)

SawyerReachEnv.reset = new_reset
env = SawyerReachEnv(renders=True, control_xyz_position_only=False)
env.reset()
pygame.init()
screen = pygame.display.set_mode((100, 100))
time = pygame.time.get_ticks()
index = 0
scaleAmount = 0.1
indexStep = 1
render_object(objects[index], scaling[json_key(index)])

while True:
    for event in pygame.event.get():
        event_happened = True
        if event.type == QUIT:
            json_dump()
            sys.exit()
        if event.type == KEYDOWN:
            if event.key == pygame.K_LEFT:
                json_dump()
                index -= indexStep
                index %= len(objects)
                while index < 0:
                    index += len(objects)
                env.reset()
                render_object(objects[index], scaling[json_key(index)])
                print('Index: ', index)
            elif event.key == pygame.K_RIGHT:
                json_dump()
                index += indexStep
                index %= len(objects)
                env.reset()
                render_object(objects[index], scaling[json_key(index)])
                print('Index: ', index)
            elif event.key == pygame.K_UP:
                scaling[json_key(index)] += scaleAmount
            elif event.key == pygame.K_DOWN:
                scaling[json_key(index)] -= scaleAmount
            else:
                pressed = chr(event.dict['key'])
                if pressed == 's':
                    json_dump() 
                elif pressed == 'r':
                    env.reset()
                    render_object(objects[index], scaling[json_key(index)])
                elif pressed == 'o':
                    scaleAmount /= 2
                    print('Scale Amount: ',scaleAmount)
                elif pressed == 'p':
                    scaleAmount *= 2
                    print('Scale Amount: ', scaleAmount)
                elif pressed == 'z':
                    print('Index: ',index)
                elif pressed == 'k':
                    if indexStep != 1:
                        indexStep //= 5 
                    print('Index Step: ', indexStep)
                elif pressed == 'l':
                    indexStep *= 5
                    print('Index Step: ', indexStep)  
                elif pressed == 'q':
                    json_dump()
                    sys.exit()
    
                          
    p.stepSimulation()
