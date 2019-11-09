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
        'id': 'SawyerLid-v0',
        'entry_point': ('roboverse.envs.sawyer_lid:SawyerLidEnv'),
    },
    {
        'id': 'SawyerSoup-v0',
        'entry_point': ('roboverse.envs.sawyer_soup:SawyerSoupEnv'),
    },
    # {
    #     'id': 'ParallelSawyerLift-v0',
    #     'entry_point': ('roboverse.envs.parallel_env:ParallelEnv'),
    #     'kwargs': {'env': 'SawyerLift-v0'},
    # },    
    # {
    #     'id': 'ParallelSawyerLid-v0',
    #     'entry_point': ('roboverse.envs.parallel_env:ParallelEnv'),
    #     'kwargs': {'env': 'SawyerLid-v0'},
    # },
)

PARALLEL_ENVIRONMENT_SPECS = tuple(
    {
        'id': 'Parallel' + env['id'],
        'entry_point': ('roboverse.envs.parallel_env:ParallelEnv'),
        'kwargs': {'env': env['id']},
    } for env in SEQUENTIAL_ENVIRONMENT_SPECS
)

BULLET_ENVIRONMENT_SPECS = SEQUENTIAL_ENVIRONMENT_SPECS + PARALLEL_ENVIRONMENT_SPECS

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