from multiworld.core.wrapper_env import ProxyEnv
class FlattenedEnv(ProxyEnv):

     def __init__(self, wrapped_env, obs_key):
        self.quick_init(locals())
        super().__init__(wrapped_env)
        self.observation_space = self.observation_space.spaces[obs_key]
        self.obs_key = obs_key


     def reset(self):
        obs = self.wrapped_env.reset()
        return obs[self.obs_key]

     def step(self, action):
        obs, reward, done, info = self.wrapped_env.step(action)
        return obs[self.obs_key], reward, done, info