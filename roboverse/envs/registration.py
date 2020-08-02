import gym

SEQUENTIAL_ENVIRONMENT_SPECS = (
    {
        'id': 'SawyerBase-v0',
        'entry_point': ('roboverse.envs.sawyer_base:SawyerBaseEnv'),
    },
    {
        'id': 'SawyerLift-v0',
        'entry_point': ('roboverse.envs.sawyer_lift:SawyerLiftEnv'),
    },
    {
        'id': 'SawyerGraspOne-v0',
        'entry_point': ('roboverse.envs.sawyer_grasp:SawyerGraspOneEnv'),
        'kwargs': {'max_force': 100, 'action_scale': 0.05}
    },
    {
        'id': 'SawyerGraspOneV2-v0',
        'entry_point': ('roboverse.envs.sawyer_grasp_v2:SawyerGraspV2Env'),
        'kwargs': {'max_force': 100,
                   'action_scale': 0.05,
                   'pos_init': [0.7, 0.2, -0.2],
                   'pos_low': [.5, -.05, -.38],
                   'pos_high': [.9, .30, -.15],
                   'object_position_low': (.65, .10, -.20),
                   'object_position_high': (.80, .25, -.20),
                   'num_objects': 1,
                   'object_ids': [1]
                   }
    },
    {
        'id': 'SawyerGraspV2-v0',
        'entry_point': ('roboverse.envs.sawyer_grasp_v2:SawyerGraspV2Env'),
        'kwargs': {'max_force': 100,
                   'action_scale': 0.05,
                   'pos_init': [0.7, 0.2, -0.2],
                   'pos_low': [.5, -.05, -.38],
                   'pos_high': [.9, .30, -.15],
                   'num_objects': 5,
                   }
    },
    {
        'id': 'SawyerGraspTenV2-v0',
        'entry_point': ('roboverse.envs.sawyer_grasp_v2:SawyerGraspV2Env'),
        'kwargs': {'max_force': 100,
                   'action_scale': 0.05,
                   'pos_init': [0.7, 0.2, -0.2],
                   'pos_low': [.5, -.05, -.38],
                   'pos_high': [.9, .30, -.15],
                   'num_objects': 10,
                   }
    },
    {
        'id': 'SawyerGraspOneV3-v0',
        'entry_point': ('roboverse.envs.sawyer_grasp_v3:SawyerGraspV3Env'),
        'kwargs': {'max_force': 100,
                   'action_scale': 0.05,
                   'pos_init': [0.7, 0.2, -0.2],
                   'pos_low': [.5, -.05, -.38],
                   'pos_high': [.9, .30, -.15],
                   'object_position_low': (.65, .10, -.20),
                   'object_position_high': (.80, .25, -.20),
                   'num_objects': 1,
                   'height_threshold': -0.3,
                   'object_ids': [1]
                   }
    },
    {
        'id': 'SawyerGraspOneV4-v0',
        'entry_point': ('roboverse.envs.sawyer_grasp_v4:SawyerGraspV4Env'),
        'kwargs': {'max_force': 100,
                   'action_scale': 0.05,
                   'pos_init': [0.7, 0.2, -0.2],
                   'pos_low': [.5, -.05, -.38],
                   'pos_high': [.9, .30, -.15],
                   'object_position_low': (.65, .10, -.20),
                   'object_position_high': (.80, .25, -.20),
                   'num_objects': 10,
                   # 'height_threshold': -0.3,
                   # 'object_ids': [1]
                   }
    },
    {
        'id': 'SawyerRigGrasp-v0',
        'entry_point': ('roboverse.envs.sawyer_rig_grasp_v0:SawyerRigGraspV0Env'),
        'kwargs': {'max_force': 100,
                   'action_scale': 0.05,
                   }
    },
    {
        'id': 'SawyerRigGR-v0',
        'entry_point': ('roboverse.envs.sawyer_rig_gr_v0:SawyerRigGRV0Env'),
        'kwargs': {'max_force': 100,
                   'action_scale': 0.05,
                   }
    },
    {
        'id': 'SawyerRigMultiobj-v0',
        'entry_point': ('roboverse.envs.sawyer_rig_multiobj_v0:SawyerRigMultiobjV0'),
        'kwargs': {'max_force': 100,
                   'action_scale': 0.05,
                   }
    },
    {
        'id': 'SawyerDistractorReaching-v0',
        'entry_point': ('roboverse.envs.disco.sawyer_distractor_reaching_v0:SawyerDistractorReachingV0'),
        'kwargs': {'max_force': 100,
                   'action_scale': 0.05,
                   }
    },
    {
        'id': 'SawyerRigVae-v0',
        'entry_point': ('roboverse.envs.sawyer_rig_vae:SawyerRigVaeEnv'),
        'kwargs': {'max_force': 100,
                   'action_scale': 0.05,
                   }
    },
    {
        'id': 'SawyerRigGrasp-v1',
        'entry_point': ('roboverse.envs.sawyer_rig_grasp_v1:SawyerRigGraspV1Env'),
        'kwargs': {'max_force': 100,
                   'action_scale': 0.05,
                   'pos_init': [0.7, 0.2, -0.2],
                   'pos_low': [.5, -.05, -.38],
                   'pos_high': [.9, .30, -.15],
                   'object_position_low': (.65, .10, -.20),
                   'object_position_high': (.80, .25, -.20),
                   'num_objects': 1,
                   # 'height_threshold': -0.3,
                   # 'object_ids': [1]
                   }
    },
    {
        'id': 'SawyerGraspOneObjectSetTenV3-v0',
        'entry_point': ('roboverse.envs.sawyer_grasp_v3:SawyerGraspV3ObjectSetEnv'),
        'kwargs': {'max_force': 100,
                   'action_scale': 0.05,
                   'pos_init': [0.7, 0.2, -0.2],
                   'pos_low': [.5, -.05, -.38],
                   'pos_high': [.9, .30, -.15],
                   'object_position_low': (.65, .10, -.20),
                   'object_position_high': (.80, .25, -.20),
                   'num_objects': 1,
                   'height_threshold': -0.3,
                   }
    },
    {
        'id': 'SawyerReach-v0',
        'entry_point': ('roboverse.envs.sawyer_reach:SawyerReachEnv'),
        'kwargs': {'max_force': 100, 'action_scale': 0.05}
    },
    {
        'id': 'SawyerLid-v0',
        'entry_point': ('roboverse.envs.sawyer_lid:SawyerLidEnv'),
    },
    {
        'id': 'SawyerSoup-v0',
        'entry_point': ('roboverse.envs.sawyer_soup:SawyerSoupEnv'),
    },
    {

        'id': 'WidowBase-v0',
        'entry_point': ('roboverse.envs.widow_base:WidowBaseEnv'),
    },
    {
        'id': 'WidowX200Grasp-v0',
        'entry_point': ('roboverse.envs.widowx200_grasp:WidowX200GraspEnv'),
        'kwargs': {'max_force': 100, 'action_scale': 0.05}
    },
    {
        'id': 'Widow200GraspV2-v0',
        'entry_point': ('roboverse.envs.widow200_grasp_v2:Widow200GraspV2Env'),
        'kwargs': {'max_force': 100, 'action_scale': 0.05}
    },

)

GRASP_V3_ENV_SPECS = []
OBJ_IDS_TEN = [0, 1, 25, 30, 50, 215, 255, 265, 300, 310]
for i, obj_id in enumerate(OBJ_IDS_TEN):
    env_params = dict(
        id='SawyerGraspOne-{}-V3-v0'.format(i),
        entry_point=('roboverse.envs.sawyer_grasp_v3:SawyerGraspV3Env'),
        kwargs={'max_force': 100,
                   'action_scale': 0.05,
                   'pos_init': [0.7, 0.2, -0.2],
                   'pos_low': [.5, -.05, -.38],
                   'pos_high': [.9, .30, -.15],
                   'object_position_low': (.65, .10, -.20),
                   'object_position_high': (.80, .25, -.20),
                   'num_objects': 1,
                   'height_threshold': -0.3,
                   'object_ids': [obj_id]
               }
    )
    GRASP_V3_ENV_SPECS.append(env_params)
GRASP_V3_ENV_SPECS = tuple(GRASP_V3_ENV_SPECS)


PROJECTION_ENVIRONMENT_SPECS = tuple(
    {
        'id': env['id'].split('-')[0] + '2d-' + env['id'].split('-')[1],
        'entry_point': ('roboverse.envs.sawyer_2d:Sawyer2dEnv'),
        'kwargs': {'env': env['id']},
    } for env in SEQUENTIAL_ENVIRONMENT_SPECS
)

PARALLEL_ENVIRONMENT_SPECS = tuple(
    {
        'id': 'Parallel' + env['id'],
        'entry_point': ('roboverse.envs.parallel_env:ParallelEnv'),
        'kwargs': {'env': env['id']},
    } for env in SEQUENTIAL_ENVIRONMENT_SPECS + PROJECTION_ENVIRONMENT_SPECS
)

BULLET_ENVIRONMENT_SPECS = SEQUENTIAL_ENVIRONMENT_SPECS + \
                           PROJECTION_ENVIRONMENT_SPECS + \
                           PARALLEL_ENVIRONMENT_SPECS + \
                           GRASP_V3_ENV_SPECS 

def register_bullet_environments():
    for bullet_environment in BULLET_ENVIRONMENT_SPECS:
        gym.register(**bullet_environment)

    gym_ids = tuple(
        environment_spec['id']
        for environment_spec in BULLET_ENVIRONMENT_SPECS)

    return gym_ids

def make(env_name, *args, **kwargs):
    env = gym.make(env_name, *args, **kwargs)
    return env
