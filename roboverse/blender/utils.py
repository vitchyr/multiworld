import os
from pprint import pprint
import numpy as np
import pdb

from collections import namedtuple

# from roboverse.blender.structs import (
# 	RevoluteJoint,
# 	FixedJoint,
# 	StationaryJoint,
# )

# RevoluteJoint = namedtuple('RevoluteJoint', ['object', 'location', 'orientation', 'low', 'high'])
# FixedJoint = namedtuple('FixedJoint', ['object', 'target'])
# StationaryJoint = namedtuple('StationaryJoint', ['object'])

class RevoluteJoint:
	def __init__(self, obj, target, orientation, low, high):
		self.obj = obj
		self.target = target
		self.orientation = orientation
		self.low = low
		self.high = high

	def __repr__(self):
		string = "RevoluteJoint('{}', '{}', '{}', {}, {})".format(
			self.obj,
			self.target,
			self.orientation,
			self.low,
			self.high
		)
		return string

	def get_mass(self):
		return 2

	def get_parent(self):
		return self.target

	def get_bullet_type(self):
		return 'JOINT_REVOLUTE'

	def get_axis(self):
		axis = {
			'x': [0,0,1],
			'y': [1,0,0],
			'z': [0,1,0],
		}[self.orientation]
		return axis

class FixedJoint:
	def __init__(self, obj, target):
		self.obj = obj
		self.target = target

	def __repr__(self):
		string = "FixedJoint('{}', '{}')".format(
			self.obj,
			self.target,
		)
		return string

	def get_mass(self):
		return 1

	def get_parent(self):
		return self.target

	def get_bullet_type(self):
		return 'JOINT_FIXED'

	def get_axis(self):
		return [0,0,0]

class StationaryJoint:
	def __init__(self, obj):
		self.obj = obj

	def __repr__(self):
		string = "StationaryJoint('{}')".format(
			self.obj,
		)
		return string

	def get_mass(self):
		return 0

	def get_parent(self):
		return None

	def get_bullet_type(self):
		return 'JOINT_FIXED'

	def get_axis(self):
		return [0,0,0]


def process_collection(save_dir, collection):
	get_collection_constraints(save_dir, collection)
	# joint = create_joint(collection, 'Limit Rotation')
	# 'Child of'
	# pdb.set_trace()

def get_collection_constraints(save_dir, collection):
	joints = {}
	for obj in collection.objects:
		constraint_type = get_object_constraint(obj)
		joint = create_joint_from_constraint(obj, constraint_type)
		joints.update(joint)
		save_obj(os.path.join(save_dir, obj.name + '.obj'), obj)

	filepath = os.path.join(save_dir, 'joints.txt')
	pprint(joints, stream=open(filepath, 'w'))
	pdb.set_trace()

def get_object_constraint(obj):
	constraint_types = obj.constraints.keys()
	num_constraints = len(constraint_types)
	if num_constraints == 0:
		return 'Stationary'
	elif num_constraints == 1:
		return constraint_types[0]
	else:
	 	raise RuntimeError(
	 		"Object {} has multiple constraints: {}".format(obj, constraint_types))

def create_joint_from_constraint(obj, constraint_type):
	print(obj)
	joint_fn = {
		'Limit Rotation': _create_revolute_joint,
		'Child Of': _create_fixed_joint,
		'Stationary': _create_stationary_joint,
	}[constraint_type]
	# constraint = obj.constraints[constraint_type]
	joint = joint_fn(obj, constraint_type)
	return joint

	# if constraint_type == 'Limit Rotation':
	# 	pdb.set_trace()
	# # joint_fn()


# def create_joint(collection, name):
# 	constrained_objects = [obj for obj in collection.all_objects
# 		if name in obj.constraints]
# 	assert len(constrained_objects) <= 1, 'Found multiple objects with {} constraint in {}: {}'.format(
# 		name, collection, constrained_objects)
# 	constrained_obj = constrained_objects[0]
# 	constraint = constrained_obj.constraints[name]
# 	for t in ['x', 'y', 'z']:
# 		low = getattr(constraint, 'min_{}'.format(t))
# 		high = getattr(constraint, 'max_{}'.format(t))
# 		print(t, low, high)
# 		if high - low != 0:
# 			limits = [low, high]
# 			orientation = t
# 			break
# 	## TODO : verify that this gives world coordinates
# 	base = list(constrained_obj.matrix_world.translation)
# 	joint = BlenderJoint(base, orientation, limits)
# 	base_obj = _add_joint_base(joint.base)
# 	collection.objects.link(base_obj)
# 	return joint

def _create_revolute_joint(obj, constraint_type):
	constraint = obj.constraints[constraint_type]
	base = list(obj.matrix_world.translation)
	## create dummy object at joint base position
	base_obj = _add_joint_base(obj.name + '_base', base)
	stationary_joint = _create_stationary_joint(base_obj)
	save_obj(os.path.join(save_dir, base_obj.name + '.obj'), base_obj)

	orientations = ['x', 'y', 'z']
	## check which orientation limits are in use
	is_orientation_active = [
		getattr(constraint, 'use_limit_{}'.format(o))
		for o in orientations]
	## revolute joints can only be defined on a single axis
	assert sum(is_orientation_active) == 1, \
		"Multiple active orientations for object {}".format(obj)
	## get active orientation
	orientation = orientations[np.argmax(is_orientation_active)]
	## get joint limits
	low = getattr(constraint, 'min_{}'.format(orientation))
	high = getattr(constraint, 'max_{}'.format(orientation))

	joint = RevoluteJoint(obj.name, base_obj.name, orientation, low, high)
	return {obj.name: joint, base_obj.name: stationary_joint[base_obj.name]}


def _create_fixed_joint(obj, constraint_type):
	constraint = obj.constraints[constraint_type]
	target = constraint.target
	joint = FixedJoint(obj.name, target.name)
	return {obj.name: joint}

def _create_stationary_joint(obj, *args, **kwargs):
	joint = StationaryJoint(obj.name)
	return {obj.name: joint}

####

def _add_joint_base(name, location):
	assert 'Sphere' not in [obj.name for obj in bpy.data.objects], \
		'Tried to add object named Sphere, but object already exists'
	# delete_by_fn(lambda obj: obj.name == 'joint_base')
	bpy.ops.mesh.primitive_uv_sphere_add(location=location)
	base_obj = bpy.data.objects['Sphere']
	base_obj.name = name
	return base_obj

def export_collection(save_dir, collection):
	description = {}
	for i, obj in enumerate(collection.all_objects):
		print(i, obj)
		if is_armature(obj):
			pass
		else:
			save_path = os.path.join(save_dir, '{}.obj'.format(i if obj.name != 'joint_base' else 'joint'))
			save_obj(save_path, obj)

			# if has_armature_parent(obj):
			# 	joint = create_rot_joint(obj.parent)
			# 	description[i] = joint
			# 	pdb.set_trace()
	pdb.set_trace()


########

def export_blend(save_dir):
	description = {}
	for i, obj in enumerate(bpy.data.objects):
		if is_armature(obj):
			pass
		else:
			save_path = os.path.join(save_dir, '{}.obj'.format(i))
			save_obj(save_path, obj)

			if has_armature_parent(obj):
				joint = create_rot_joint(obj.parent)
				description[i] = joint
				pdb.set_trace()

def save_obj(filepath, obj):
	select(obj)
	bpy.ops.export_scene.obj(filepath=filepath, use_selection=True)

def deselect_all():
	for obj in bpy.data.objects:
		obj.select_set(False)

def select(obj):
	deselect_all()
	obj.select_set(True)

def select_by_fn(fn):
	deselect_all()
	for obj in bpy.data.objects:
		if fn(obj):
			obj.select_set(True)

def delete_by_fn(fn):
	select_by_fn(fn)
	bpy.ops.object.delete()

def is_armature(obj):
	return 'Armature' in obj.name

def has_armature_parent(obj):
	parent = obj.parent
	return parent and 'Armature' in parent.name

def create_rot_joint(armature):
	bones = armature.pose.bones
	# pdb.set_trace()
	assert len(bones) == 1, 'Armature {} has {} bones'.format(
		armature.name, len(bones)) 
	bone = bones[0]
	rot_constraints = bone.constraints['Limit Rotation']
	for t in ['x', 'y', 'z']:
		low = getattr(rot_constraints, 'min_{}'.format(t))
		high = getattr(rot_constraints, 'max_{}'.format(t))
		if high - low != 0:
			print('got limits', t, low, high)
			limits = [low, high]
			orientation = t
			break

	base = list(armature.location)
	joint = BlenderJoint(base, orientation, limits)
	return joint


if __name__ == '__main__':
	import bpy

	save_dir = 'dump/counters_collections/'
	if not os.path.exists(save_dir):
		os.mkdir(save_dir)

	coll = bpy.data.collections[5]
	process_collection(save_dir, coll)
	pdb.set_trace()
	process_collection(bpy.data.collections[4])
	export_collection(save_dir, bpy.data.collections[4])
	pdb.set_trace()

	export_blend(save_dir)

	# path = 2
	# read_blend(path)
	pdb.set_trace()



