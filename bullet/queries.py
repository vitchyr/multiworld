import pybullet as p

from bullet.misc import quat_to_deg

########################
#### bullet queries ####
########################

def get_joint_state(body, joint, keys=None):
    lookup_fn = p.getJointState
    labels = ['pos', 'vel', 'forces', 'torque']
    return _lookup_by_joint(body, joint, lookup_fn, labels, keys)

def get_joint_info(body, joint, keys=None):
    lookup_fn = p.getJointInfo
    labels = ['joint_index', 'joint_name', 'joint_type', 'q_index', 'u_index', 'flags',
              'damping', 'friction', 'low', 'high', 'max_force', 'max_velocity',
              'link_name', 'axis', 'parent_frame_pos', 'parent_frame_theta', 'parent_index']
    return _lookup_by_joint(body, joint, lookup_fn, labels, keys)

def get_link_state(body, link, keys=None):
    lookup_fn = p.getLinkState
    labels = ['pos', 'theta', 'local_inertial_pos', 'local_inertial_theta', 
              'world_link_pos', 'world_link_theta', 'world_link_linear_vel', 'world_link_angular_vel']
    return _lookup_by_joint(body, link, lookup_fn, labels, keys)

def _lookup_by_joint(body, joint, lookup_fn, labels, keys):
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
    filtered_d = filter_quat_to_deg(filtered_d)
    ####
    if len(keys) == 1:
        key = keys[0]
        return filtered_d[key]
    else:
        return filtered_d
    return filtered_d

def get_index_by_attribute(body, attr, val):
    num_joints = p.getNumJoints(body)
    # link_names = {get_link_name_from_joint(body, j): j for j in range(num_joints)}
    link_names = {get_joint_info(body, j, attr): j for j in range(num_joints)}
    link_index = link_names[val]
    return link_index

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

