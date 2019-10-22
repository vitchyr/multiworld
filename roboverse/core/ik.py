import numpy as np
import pybullet as p
import pdb

from roboverse.core.queries import (
    get_joint_info,
    get_joint_state,
    get_link_state,
)

from roboverse.core.misc import (
    quat_to_deg,
    l2_dist,
    rot_diff_deg,
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

def ee_approx_eq(pos_a, theta_a, pos_b, theta_b, pos_eps=1e-5, theta_eps=1):
    '''
        theta_a and theta_b : euler angles in degrees
        returns True if ||pos_a-pos_b||_2 <= pos_eps
        and ||theta_a-theta_b||_2 <= theta_eps
    '''
    pos_delta = l2_dist(pos_a, pos_b)
    theta_delta = rot_diff_deg(theta_a, theta_b)
    return pos_delta <= pos_eps and theta_delta <= theta_eps

def get_num_actuators(body):
    '''
        TODO : there is probably a better way to do this
    '''
    joint_indices, _ = get_joint_positions(body)
    return len(joint_indices)

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
              gripper_bounds=(0,1), arm_vel_mult=3, gripper_vel_mult=10, discrete_gripper=True):
    gripper_state = get_gripper_state(body, gripper, gripper_bounds, discrete_gripper)
    #### ik
    ik_solution = ik(body, link, pos, theta, damping)
    ik_solution[-2:] = gripper_state
    #### velocities
    joints, velocities = ik_to_joint_vel(body, ik_solution)
    velocities[:-2] *= arm_vel_mult
    velocities[-2:] *= gripper_vel_mult
    #### check if end effector already at correct position and orientation
    link_pos, link_deg = get_link_state(body, link, ['pos', 'theta'], return_list=True)
    deg = quat_to_deg(theta)
    if ee_approx_eq(link_pos, link_deg, pos, deg):
        velocities[:-2] = 0
    #### apply velocities
    velocity_control(body, joints, velocities)

def step_ik(body=0):
    '''
        enforces joint limits for gripper fingers
    '''
    p.stepSimulation()
    for joint in range(20, 25):
        low, high = get_joint_info(body, joint, ['low', 'high'], return_list=True)
        pos = get_joint_state(body, joint, 'pos')
        pos = np.clip(pos, low, high)
        p.resetJointState(body, joint, pos)

## gripper

def get_gripper_state(body, gripper, gripper_bounds, discrete_gripper):
    l_limits, r_limits = _get_gripper_limits(body)
    if discrete_gripper:
        return _get_discrete_gripper_state(gripper, gripper_bounds, l_limits, r_limits)
    else:
        return _get_continuous_gripper_state(gripper, gripper_bounds, l_limits, r_limits)

def _get_gripper_limits(body):
    l_limits = get_joint_info(body, 'right_gripper_l_finger_joint',
                              ['low', 'high'])
    r_limits = get_joint_info(body, 'right_gripper_r_finger_joint',
                              ['low', 'high'])
    return l_limits, r_limits

def _get_discrete_gripper_state(gripper, gripper_bounds, l_limits, r_limits):
    low, high = gripper_bounds
    gripper_close_thresh = (low + high) / 2.
    if gripper > gripper_close_thresh:
        ## close gripper
        gripper_state = [l_limits['low'], r_limits['high']]
    else:
        ## open gripper
        gripper_state = [l_limits['high'], r_limits['low']]
    return gripper_state

def _get_continuous_gripper_state(gripper, gripper_bounds, l_limits, r_limits):
    low, high = gripper_bounds
    percent_closed = (gripper - low) / (high - low)
    l_state = l_limits['high'] + percent_closed * (l_limits['low'] - l_limits['high'])
    r_state = r_limits['low'] + percent_closed * (r_limits['high'] - r_limits['low'])
    return [l_state, r_state]


