import pybullet as p
import pdb

from roboverse.bullet.misc import deg_to_quat

def set_body_state(body, pos, deg):
	quat = deg_to_quat(deg)
	p.resetBasePositionAndOrientation(body, pos, quat)