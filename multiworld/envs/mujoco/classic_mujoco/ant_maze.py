import numpy as np

from multiworld.envs.mujoco.classic_mujoco.ant import AntEnv


class AntMazeEnv(AntEnv):
    def __init__(
            self,
            *args,
            model_path='classic_mujoco/ant_maze.xml',
            **kwargs
    ):
        self.quick_init(locals())
        super().__init__(
            *args,
            model_path=model_path,
            **kwargs
        )

    def _sample_random_goal_vectors(self, batch_size):
        assert self.goal_is_xy
        goals = np.random.uniform(
            self.goal_space.low,
            self.goal_space.high,
            size=(batch_size, self.goal_space.low.size),
        )
        if self.two_frames:
            goals = goals[:,:int(self.goal_space.low.size/2)]

        if self.goal_is_xy:
            goals[(0 <= goals) * (goals < 0.5)] += 2
            goals[(0 <= goals) * (goals < 1.5)] += 1.5
            goals[(0 >= goals) * (goals > -0.5)] -= 2
            goals[(0 >= goals) * (goals > -1.5)] -= 1.5
        return goals


if __name__ == '__main__':
    env = AntMazeEnv(
        goal_low=[-4, -4],
        goal_high=[4, 4],
        goal_is_xy=True,
        reward_type='xy_dense',
    )
    import gym
    from multiworld.envs.mujoco import register_custom_envs
    register_custom_envs()
    # env = gym.make('AntMaze150Env-v0')
    env = gym.make('AntCrossMaze150Env-v0')
    env = gym.make('DebugAntMaze90BottomLeftRandomizeInitEnv-v0')
    env.reset()
    i = 0
    while True:
        i += 1
        env.render()
        action = env.action_space.sample()
        # action = np.zeros_like(action)
        _, reward, *_ = env.step(action)
        # print(reward, np.linalg.norm(env.sim.data.get_body_xpos('torso')[:2]
        #                              - env._xy_goal) )
        # print(env.sim.data.qpos)
        if i % 50 == 0:
            env.reset()
