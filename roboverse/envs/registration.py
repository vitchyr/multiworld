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