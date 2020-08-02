import os
import pdb

import pybullet as p
import pybullet_data as pdata
import math

from roboverse.bullet.misc import (
  load_urdf,
  deg_to_quat,
)

def loader(*filepath, **defaults):
    filepath = os.path.join(*filepath)
    def fn(*args, **kwargs):
        defaults.update(kwargs)

        if 'deg' in defaults:
          assert 'quat' not in defaults
          defaults['quat'] = deg_to_quat(defaults['deg'])
          del defaults['deg']
        return load_urdf(filepath, **defaults)
    return fn

cur_path = os.path.dirname(os.path.realpath(__file__))
ASSET_PATH = os.path.join(cur_path, '../envs/assets')
PDATA_PATH = pdata.getDataPath()
obj_dir = "bullet-objects"

## robots

sawyer = loader(ASSET_PATH, 'sawyer_robot/sawyer_description/urdf/sawyer_xacro.urdf')
sawyer_invisible = loader(ASSET_PATH, 'sawyer_robot/sawyer_description/urdf/sawyer_xacro_invisible.urdf')
sawyer_finger_visual_only = loader(ASSET_PATH, 'sawyer_robot/sawyer_description/urdf/sawyer_xacro_finger_visual_only.urdf')
sawyer_hand_visual_only = loader(ASSET_PATH, 'sawyer_robot/sawyer_description/urdf/simple_sawyer_xacro_finger_visual_only.urdf')
widowx_200 = loader(
  ASSET_PATH,
  'interbotix_descriptions/urdf/wx200.urdf',
  pos=[0.6, 0, -0.4],
  deg=[math.pi, math.pi, math.pi],
  scale=1
) #pos=[0.4, 0, -0.4], quat=[0, 0, -0.707, -0.707]
#pos=[0.7, 0, 0.1]


## pybullet_data objects

table = loader(PDATA_PATH, 'table/table.urdf',
               pos=[.75, -.2, -1],
               quat=[0, 0, 0.707107, 0.707107],
               scale=1.0)

duck = loader(PDATA_PATH, 'duck_vhacd.urdf',
              pos=[.75, -.4, -.3],
              quat=[0, 0, 1, 0],
              #deg=[0,0,0],
              scale=0.8)

lego = loader(PDATA_PATH, 'lego/lego.urdf',
              pos=[.75, .2, -.3],
              quat=[0, 0, 1, 0],
              rgba=[1, 0, 0, 1],
              scale=1.1) #1.2


## custom objects

bowl = loader(ASSET_PATH, os.path.join(obj_dir, "bowl", "bowl.urdf"),
              pos=[.75, 0, -.3],
              scale=0.25)

lid = loader(ASSET_PATH, os.path.join(obj_dir, "bowl", "lid.urdf"),
              pos=[.75, 0, -.3],
              scale=0.1) #0.25

cube = loader(ASSET_PATH, os.path.join(obj_dir, "cube", "cube.urdf"),
              pos=[.75, -.4, -.3],
              quat=[0, 0, 0, 1],
              scale=0.03) #0.05

spam = loader(ASSET_PATH, os.path.join(obj_dir, "spam", "spam.urdf"),
              pos=[.75, -.4, -.3],
              #deg=[90,0,0], #90,0,-90
              quat=[0, 0, 0, 1],
              scale=0.015) #0.0175 #0.25
## tray

tray = loader('', os.path.join("tray", "tray.urdf"),
              pos=[0.70, 0.15, -0.36],
              deg=[0, 0, 0],
              scale=0.75)

box = loader(ASSET_PATH, os.path.join(obj_dir, "box", "box.urdf"),
                # pos=[0.85, 0, -.35],
                pos=[0.8, 0.075, -.35],
                scale=0.125)

widow200_tray = loader(ASSET_PATH, os.path.join(obj_dir, "tray", "tray.urdf"),
              pos=[0.8, -0.05, -0.36],
              deg=[0, 0, 0],
              scale=0.5)
bowl_sliding = loader(ASSET_PATH, 'objects/bowl_sliding/bowl.urdf',
              pos=[.75, 0, -.3],
              scale=0.25)

