import numpy as np
import pybullet as p

from roboverse.core.queries import (
    get_joint_info,
    get_joint_state,
)


############################
#### inverse kinematics ####
############################

def ik(body, link, pos, theta, damping):
    if type(damping) == float:
        ## if damping is a scalar, repeat for each degree of freedom
        n_dof = p.getNumJoints(body)
        damping = [damping for _ in range(n_dof)]
    ik_solution = p.calculateInverseKinematics(body, link, pos,
                                               targetOrientation=theta,
                                               jointDamping=damping)
    return np.array(ik_solution)


def get_joint_positions(body):
    num_joints = p.getNumJoints(body)
    q_indices = [get_joint_info(body, j, 'q_index') for j in range(num_joints)]
    joint_indices = [j for j in range(num_joints) if q_indices[j] > -1]
    joint_positions = [get_joint_state(body, j, 'pos') for j in joint_indices]
    return np.array(joint_indices), np.array(joint_positions)


def ik_to_joint_vel(body, ik_solution):
    indices, current = get_joint_positions(body)
    velocities = ik_solution - current
    return indices, velocities


def velocity_control(body, joints, velocities):
    '''
        body : int
        joints : np.ndarray of ints
        velocities : np.ndarray of floats
    '''
    joints = joints.tolist()
    velocities = velocities.tolist()
    p.setJointMotorControlArray(body, joints, p.VELOCITY_CONTROL,
                                targetVelocities=velocities)


def position_control(body, link, pos, theta, damping=1e-3):
    ik_solution = ik(body, link, pos, theta, damping)
    joint_indices, _ = get_joint_positions(body)
    for joint_ind, pos in zip(joint_indices, ik_solution):
        p.resetJointState(body, joint_ind, pos)


def sawyer_ik(body, link, pos, theta, gripper, damping=1e-3,
              gripper_close_thresh=0.5, gripper_vel_mult=20):
    #### get gripper state position
    l_limits = get_joint_info(body, 'right_gripper_l_finger_joint',
                              ['low', 'high'])
    r_limits = get_joint_info(body, 'right_gripper_r_finger_joint',
                              ['low', 'high'])
    if gripper > gripper_close_thresh:
        gripper_state = [l_limits['low'], r_limits['high']]
    else:
        gripper_state = [l_limits['high'], r_limits['low']]
    #### ik
    ik_solution = ik(body, link, pos, theta, damping)
    ik_solution[-2:] = gripper_state
    #### velocities
    joints, velocities = ik_to_joint_vel(body, ik_solution)
    velocities[-2:] *= gripper_vel_mult
    velocity_control(body, joints, velocities)
