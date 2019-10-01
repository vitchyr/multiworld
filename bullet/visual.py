import pybullet as p

from bullet.queries import (
    get_joint_info,
    get_link_state,
)

from bullet.misc import quat_to_deg

# def get_link_state_dict(body, link):
#     labels = [  'world_pos', 'world_theta', 
#                 'local_inertial_pos', 'local_inertial_theta',
#                 'world_link_frame_pos', 'world_link_theta']
#     state = p.getLinkState(body, link)
#     state = [s if ind % 2 == 0 else quat_to_deg(s) for ind, s in enumerate(state)]
#     state = {labels[ind]: state[ind] for ind in range(len(state))}
#     return state

def get_visual_state_dict(body):
    ## p.getVisualShapeData returns
    ## 0 : object_id,   1 : link_index, 2 : geometry_type
    ## 3 : dimensions   4 : filename,   5 : local_pos
    ## 6 : local_theta, 7 : rgba,       8 : texture_id
    # lookup_fn = p.getLinkState
    # labels = ['object_id', 'link_index', 'geometry_type', 'dimensions', 'filename',
    #           'visual_pos', 'visual_theta', 'rgba', 'texture_id']
    visual_data = p.getVisualShapeData(body)
    ## {link_index : {pos, theta}, ... }
    visual_data = {k[1]: {  'visual_pos': k[5], 
                            'visual_theta': quat_to_deg(k[6]), 
                            'filename': k[4].decode().replace('STL', 'DAE') 
                         } for k in visual_data}
    return visual_data

def export_render_data(body=0):
    '''
        sawyer : body id of sawyer
        returns : inputs to `setup_sawyer` for rendering in Blender 2.8
    '''
    render_data = {}
    visual_data = get_visual_state_dict(body)

    for link, data in visual_data.items():
        link_name = get_joint_info(body, link, 'link_name')
        # link_name = get_link_name(body, link)
        link_state = get_link_state(body, link)
        render_data[link_name] = link_state
        render_data[link_name].update(visual_data[link])
    
    render_data = {k: v for k, v in render_data.items() if len(v['filename'])}
    return render_data