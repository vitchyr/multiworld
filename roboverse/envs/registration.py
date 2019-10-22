import gym

BULLET_ENVIRONMENT_SPECS = (
    {
        'id': 'SawyerLift-v0',
        'entry_point': ('roboverse.envs.sawyer_lift:SawyerLiftEnv'),
    },
)

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