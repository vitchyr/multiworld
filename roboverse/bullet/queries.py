import pdb
import numpy as np
import pybullet as p

from roboverse.bullet.misc import quat_to_deg


########################
#### bullet queries ####
########################

def get_joint_state(body, joint, keys=None, return_list=False, quat_to_deg=True):
    lookup_fn = p.getJointState
    labels = ['pos', 'vel', 'forces', 'torque']
    return _lookup_by_joint(body, joint, lookup_fn, labels, keys, return_list, quat_to_deg)


def get_joint_info(body, joint, keys=None, return_list=False, quat_to_deg=True):
    lookup_fn = p.getJointInfo
    labels = ['joint_index', 'joint_name', 'joint_type', 'q_index', 'u_index', 'flags',
              'damping', 'friction', 'low', 'high', 'max_force', 'max_velocity',
              'link_name', 'axis', 'parent_frame_pos', 'parent_frame_theta', 'parent_index']
    return _lookup_by_joint(body, joint, lookup_fn, labels, keys, return_list, quat_to_deg)


def get_link_state(body, link, keys=None, return_list=False, quat_to_deg=True):
    lookup_fn = p.getLinkState
    labels = ['pos', 'theta', 'local_inertial_pos', 'local_inertial_theta',
              'world_link_pos', 'world_link_theta', 'world_link_linear_vel', 'world_link_angular_vel']
    return _lookup_by_joint(body, link, lookup_fn, labels, keys, return_list, quat_to_deg)

def get_body_info(body, keys=None, return_list=False, quat_to_deg=True):
    lookup_fn = lambda body, joint: p.getBasePositionAndOrientation(body)
    labels = ['pos', 'theta']
    joint = None
    return _lookup_by_joint(body, joint, lookup_fn, labels, keys, return_list, quat_to_deg)

def _lookup_by_joint(body, joint, lookup_fn, labels, keys, return_list, quat_to_deg):
    keys = keys or labels
    ## get joint index from name if joint is not already an index
    joint = coerce_to_joint_name(body, joint)
    ## convert keys to a singleton list if keys is not already a list
    keys = coerce_to_list(keys)
    ## ensure all query keys are in labels for this lookup function
    assert all([key in labels for key in keys])
    ## bullet lookup functions returns list of vals
    raw_info = lookup_fn(body, joint)
    ## turn raw bullet output into a dictionary with key labels
    info_d = {labels[i]: raw_info[i] for i in range(len(raw_info))}
    ## filter dictionary by query keys
    filtered_d = {k: v for k, v in info_d.items() if k in keys}
    ## turn any bytes vals into strings (usually `joint_name` or `filename`)
    filtered_d = filter_bytes_to_str(filtered_d)
    ## turn any quaternions into euler angles in degrees
    if quat_to_deg: filtered_d = filter_quat_to_deg(filtered_d)
    ####
    output = format_query(filtered_d, keys, return_list)
    return output

def format_query(filtered_d, keys, return_list):
    if len(keys) == 1:
        ## return val for sole key
        key = keys[0]
        return filtered_d[key]
    elif return_list:
        ## return list of vals
        return [filtered_d[k] for k in keys]
    else:
        ## return dictionary
        return filtered_d

def get_index_by_attribute(body, attr, val):
    num_joints = p.getNumJoints(body)
    link_names = {get_joint_info(body, j, attr): j for j in range(num_joints)}
    link_index = link_names[val]
    return link_index

#########################
#### gym env queries ####
#########################

def format_sim_query(bodies, links, joints):
    body_queries = bodies

    link_queries = links

    joint_queries = []
    for body, joint in  joints:
        if joint is None:
            num_joints = p.getNumJoints(body)
            joint_queries.extend([(body, joint) for joint in range(num_joints)])
        else:
            joint = coerce_to_joint_name(body, joint)
            joint_queries.append((body, joint))

    return body_queries, link_queries, joint_queries

def get_sim_state(body_queries=None, link_queries=None, joint_queries=None):
    sim_state = []

    ## body queries
    for body in body_queries:
        pos, theta = get_body_info(body, return_list=True, quat_to_deg=False)
        sim_state.extend(pos)
        sim_state.extend(theta)
        # print('body: ', body, pos, theta)

    ## link queries
    for body, link in link_queries:
        pos, theta = get_link_state(body, link, ['pos', 'theta'], return_list=True, quat_to_deg=False)
        sim_state.extend(pos)
        sim_state.extend(theta)
        # print('link: ', body, link, pos, theta)

    ## joint queries
    for body, joint in joint_queries:
        pos, vel = get_joint_state(body, joint, ['pos', 'vel'], return_list=True, quat_to_deg=False)
        sim_state.append(pos)
        sim_state.append(vel)
        # print('joint: ', body, joint, pos, vel)

    sim_state = np.array(sim_state)
    return sim_state

def has_fixed_root(body):
    if p.getNumJoints(body) == 0:
        return False
    else:
        joint_name, joint_type = get_joint_info(body, 0, ['joint_name', 'joint_type'], return_list=True)
        return joint_name == 'base_joint' and joint_type == p.JOINT_FIXED

##########################
#### helper functions ####
##########################

def coerce_to_joint_name(body, joint):
    if type(joint) == str:
        return get_index_by_attribute(body, 'joint_name', joint)
    else:
        return joint


def coerce_to_list(l):
    if type(l) == list:
        return l
    else:
        return [l]


def filter_bytes_to_str(dictionary):
    '''
        input : dict
    '''
    for k, v in dictionary.items():
        if type(v) == bytes:
            dictionary[k] = v.decode()
    return dictionary


def filter_quat_to_deg(dictionary):
    for k, v in dictionary.items():
        if 'theta' in k and len(v) == 4:
            dictionary[k] = quat_to_deg(v)
    return dictionary
