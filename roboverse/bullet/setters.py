import pybullet as p

def set_body_state(body, pos, quat):
	p.resetBasePositionAndOrientation(body, pos, quat)