import os
import time
import numpy as np
import pdb

import roboverse.bullet as bullet
import roboverse.devices as devices

from roboverse.blender.utils import (
	RevoluteJoint,
	FixedJoint,
	StationaryJoint,
)

# space_mouse = devices.SpaceMouse()
# space_mouse.start_control()

bullet.connect()
bullet.setup()


####
# import pybullet as p
# base = flag

# collision_flag = p.createCollisionShape()
# visual_flag = p.createVisualShape()

# linkMasses = [1]
# linkCollisionShapeIndices = [collision_flag]
# linkVisualShapeIndices = [visual_flag]
# linkParentIndices = [0]
# linkJointTypes = [p.JOINT_REVOLUTE]

####

## load meshes
# sawyer = bullet.objects.sawyer(quat=[0,0,1,0])
# table = bullet.objects.table()
# spam = bullet.objects.spam()

# bullet.load_obj()
import pybullet as p

folder = 'dump/counters_collections/'
meshes = [file.replace('.obj', '') for file in os.listdir(folder) if 'obj' in file]
joints = eval(open(os.path.join(folder, 'joints.txt'), 'r').read())
num_meshes = len(meshes)

scale = 0.075

collision_shape_map = {
	mesh: p.createCollisionShape(
		p.GEOM_MESH,
		fileName=os.path.join(folder, mesh + '.obj'),
		meshScale=[scale,scale,scale])
	for mesh in meshes
}
visual_shape_map = {
	mesh: p.createVisualShape(
		p.GEOM_MESH,
		fileName=os.path.join(folder, mesh + '.obj'),
		meshScale=[scale,scale,scale])
	for mesh in meshes
}

primitive_collision_shape = p.createCollisionShape(
		p.GEOM_BOX,
		halfExtents=[.1,.1,.1])
primitve_visual_shape = p.createVisualShape(
		p.GEOM_BOX,
		halfExtents=[.1,.1,.1])

primitive_collision_shape2 = p.createCollisionShape(
		p.GEOM_BOX,
		halfExtents=[.01,.2,.2])
primitve_visual_shape2 = p.createVisualShape(
		p.GEOM_BOX,
		halfExtents=[.01,.2,.2])

base_mesh = meshes[0]
link_meshes = meshes[5:6]

# collision_shapes = [
# 	collision_shape_map[mesh]
# 	for mesh in link_meshes
# ]
# visual_shapes = [
# 	visual_shape_map[mesh]
# 	for mesh in link_meshes
# ]
collision_shapes = [primitive_collision_shape2]
visual_shapes = [primitve_visual_shape2]


print(collision_shapes, visual_shapes)

link_masses = [
	joints[mesh].get_mass()
	for mesh in link_meshes
]

parent_meshes = [
	joints[mesh].get_parent()
	for mesh in link_meshes
]
link_parents = [
	collision_shape_map[mesh] if type(mesh) is str else -1
	for mesh in parent_meshes
]

joint_type_names = [
	joints[mesh].get_bullet_type()
	for mesh in link_meshes
]
joint_types = [
	getattr(p, name)
	for name in joint_type_names
]

joint_axes = [
	joints[mesh].get_axis()
	for mesh in link_meshes
]

link_positions = [[0,0,.2]]*len(parent_meshes)
link_orientations = [[0,0,0,1]]*len(parent_meshes)
intertial_positions = [[0,0,0]]*len(parent_meshes)
inertial_orientations = [[0,0,0,1]]*len(parent_meshes)

base_mass = 0
base_shape = collision_shape_map[base_mesh]
base_position = [0,0,0] #[-.75, 0, -.75]
# base_orientation = [.707,0,0,.707]
base_orientation=[0,0,0,1]
# link_masses = [.1]*len(collision_shapes)
# link_parents = [0] + [1]*(len(collision_shapes)-1)
# link_positions = [[0, 0, 0]]*len(collision_shapes)
# link_orientations = [[0,0,0,1]]*len(collision_shapes)
# intertial_positions = [[0, 0, 0]]*len(collision_shapes)
# inertial_orientations = [[0, 0, 0, 1]]*len(collision_shapes)
# joint_types = [p.JOINT_REVOLUTE] + [p.JOINT_FIXED]*(len(collision_shapes)-1)
# joint_axes = [[0,1,0]]*len(collision_shapes)

body = p.createMultiBody(base_mass,
	primitive_collision_shape,
	basePosition=base_position,
	baseOrientation=base_orientation,
	linkMasses=link_masses,
	linkCollisionShapeIndices=collision_shapes, 
	linkVisualShapeIndices=visual_shapes,
	linkPositions=link_positions,
	linkOrientations=link_orientations,
	linkInertialFramePositions=intertial_positions,
	linkInertialFrameOrientations=inertial_orientations,
	linkParentIndices=link_parents,
	linkJointTypes=joint_types,
	linkJointAxis=joint_axes)

print(
	'base_mass: {}\n'.format(base_mass),
	'base_shape: {}\n'.format(primitive_collision_shape),
	'base_position: {}\n'.format(base_position),
	'base_orientation: {}\n'.format(base_orientation),
	'link_masses: {}\n'.format(link_masses),
	'collision_shapes: {}\n'.format(collision_shapes),
	'visual_shapes: {}\n'.format(visual_shapes),
	'link_positions: {}\n'.format(link_positions),
	'link_orientations: {}\n'.format(link_orientations),
	'inertial_orientations: {}\n'.format(inertial_orientations),
	'link_parents: {}\n'.format(link_parents),
	'joint_types: {}\n'.format(joint_types),
	'joint_axes: {}\n'.format(joint_axes)
	)

# pdb.set_trace()

# import pybullet as p
# collision_shapes = []
# visual_shapes = []
# for i in range(6):
# 	filename = 'dump/counters_collections/{}.obj'.format(i)
# 	print(filename)
# 	scale = 0.075
# 	collision_shape = p.createCollisionShape(p.GEOM_MESH, fileName=filename, meshScale=[scale,scale,scale])
# 	visual_shape = p.createVisualShape(p.GEOM_MESH, fileName=filename, meshScale=[scale,scale,scale])
# 	collision_shapes.append(collision_shape)
# 	visual_shapes.append(visual_shape)	

# base_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=.1)
# base_mass = 0
# # base_position = np.array([2.5, -5.900000095367432, 16.750001907348633]) * scale + np.array([-.75, 0, -.75])
# base_position = [-.75, 0, -.75]
# base_orientation = [.707,0,0,.707]
# link_masses = [.1]*len(collision_shapes)
# link_parents = [0] + [1]*(len(collision_shapes)-1)
# link_positions = [[0, 0, 0]]*len(collision_shapes)
# link_orientations = [[0,0,0,1]]*len(collision_shapes)
# intertial_positions = [[0, 0, 0]]*len(collision_shapes)
# inertial_orientations = [[0, 0, 0, 1]]*len(collision_shapes)
# joint_types = [p.JOINT_REVOLUTE] + [p.JOINT_FIXED]*(len(collision_shapes)-1)
# joint_axes = [[0,1,0]]*len(collision_shapes)
# # p.createMultiBody(base_mass, base_shape)
# print(collision_shapes)
# print(visual_shapes)
# print(link_masses)
# body = p.createMultiBody(base_mass,
# 	base_shape,
# 	basePosition=base_position,
# 	baseOrientation=base_orientation,
# 	linkMasses=link_masses,
# 	linkCollisionShapeIndices=collision_shapes, 
# 	linkVisualShapeIndices=visual_shapes,
# 	linkPositions=link_positions,
# 	linkOrientations=link_orientations,
# 	linkInertialFramePositions=intertial_positions,
# 	linkInertialFrameOrientations=inertial_orientations,
# 	linkParentIndices=link_parents,
# 	linkJointTypes=joint_types,
# 	linkJointAxis=joint_axes)
# # p.createMultiBody()
# pdb.set_trace()

# p.createMultiBody(
# 	base_shape_2,
# 	parent=body,

# )
# mass = 0
# p.createMultiBody(mass, collision_shape, visual_shape, baseOrientation=[.707,0,0,.707], basePosition=[-.75,0,-.75])

while True:
    time.sleep(0.01)
    bullet.step()
    # pdb.set_trace()
