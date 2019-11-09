import numpy as np
import pybullet as p

from roboverse.bullet import (
  get_bbox,
  bbox_intersecting,
)

class Sensor:

  def __init__(self, body, xyz_min=[.7, .4, -.3], xyz_max=[.8, .5, -.2], visualize=False, rgba=[1,1,0,.5]):
    self._body = body
    self._xyz_min = np.array(xyz_min)
    self._xyz_max = np.array(xyz_max)
    self._rgba = rgba

    self._bbox = (self._xyz_min, self._xyz_max)
    self._half_extents = (self._xyz_max - self._xyz_min) / 2.
    self._base_position = (self._xyz_min + self._xyz_max) / 2.
    if visualize:
      self._create_visual_shape()

  def get_pos(self):
    return self._base_position
  def _create_visual_shape(self):
    visual_shape = p.createVisualShape(shapeType=p.GEOM_BOX,
                      rgbaColor=self._rgba,
                      halfExtents=self._half_extents)
    body = p.createMultiBody(baseMass=0,
                      baseVisualShapeIndex=visual_shape,
                      basePosition=self._base_position)

  def sense(self):
    body_bbox = get_bbox(self._body)
    intersecting = bbox_intersecting(body_bbox, self._bbox)
    return int(intersecting)


