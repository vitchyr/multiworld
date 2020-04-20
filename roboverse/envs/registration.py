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
)

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

BULLET_ENVIRONMENT_SPECS = SEQUENTIAL_ENVIRONMENT_SPECS + PROJECTION_ENVIRONMENT_SPECS + PARALLEL_ENVIRONMENT_SPECS

def register_bullet_environments():
    for bullet_environment in BULLET_ENVIRONMENT_SPECS:
        gym.register(**bullet_environment)

    gym_ids = tuple(
        environment_spec['id']
        for environment_spec in  BULLET_ENVIRONMENT_SPECS)

    return gym_ids

def make(env_name, *args, **kwargs):
    env = gym.make(env_name, *args, **kwargs)
    return env
