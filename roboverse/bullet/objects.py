import os
import pdb

import pybullet as p
import pybullet_data as pdata

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


## robots

sawyer = loader(ASSET_PATH, 'sawyer_robot/sawyer_description/urdf/sawyer_xacro.urdf')


## pybullet_data objects

table = loader(PDATA_PATH, 'table/table.urdf',
               pos=[.75, -.2, -1],
               quat=[0, 0, 0.707107, 0.707107],
               scale=1.0)

duck = loader(PDATA_PATH, 'duck_vhacd.urdf',
              pos=[.75, -.4, -.3],
              deg=[0,0,0],
              scale=0.8)

lego = loader(PDATA_PATH, 'lego/lego.urdf',
              pos=[.75, .2, -.3],
              quat=[0, 0, 1, 0],
              rgba=[1, 0, 0, 1],
              scale=1.2)


## custom objects

bowl = loader(ASSET_PATH, 'objects/bowl/bowl.urdf',
              pos=[.75, 0, -.3],
              scale=0.25)

lid = loader(ASSET_PATH, 'objects/bowl/lid.urdf',
              pos=[.75, 0, -.3],
              scale=0.25)

cube = loader(ASSET_PATH, 'objects/cube/cube.urdf',
              pos=[.75, -.4, -.3],
              scale=0.05)

spam = loader(ASSET_PATH, 'objects/spam/spam.urdf',
              pos=[.75, -.4, -.3],
              deg=[90,0,-90],
              scale=0.025)

# sensor = loader(ASSET_PATH, 'objects/sensor/sensor.urdf',
#               pos=[.75, .4, -.3],
#               scale=0.05)
