import time
import roboverse.bullet as bullet
import pybullet as p

bullet.connect()
bullet.setup()

base_mass = 0
base_shape = p.createCollisionShape(
		p.GEOM_BOX,
		halfExtents=[.02,.2,.02])
base_position = [0,0,0]
base_orientation = [0,0,0,1]

num_children = 1
link_masses = [1 for _ in range(num_children)]
collision_shapes = [
	p.createCollisionShape(
		p.GEOM_BOX,
		halfExtents=[.02,.2,.4])
	for _ in range(num_children)
]
visual_shapes = [
	p.createVisualShape(
		p.GEOM_BOX,
		halfExtents=[.02,.2,.4])
	for _ in range(num_children)
]
print(collision_shapes, visual_shapes)

link_positions = [[0,0,.41]]
link_orientations = [[0,0,0,1]]
intertial_positions = [[1,1,1]]
inertial_orientations = [[0,0,0,1]]
link_parents = [0]
joint_types = [p.JOINT_REVOLUTE]
joint_axes = [[0,1,0]]

body = p.createMultiBody(base_mass,
	base_shape,
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
	'base_shape: {}\n'.format(base_shape),
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

while True:
    time.sleep(0.01)
    bullet.step()