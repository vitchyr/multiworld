import numpy as np

from multiworld.envs.mujoco.classic_mujoco.ant import AntEnv


class AntMazeEnv(AntEnv):

    def _collision_idx(self, pos):
        bad_pos_idx = []
        for i in range(len(pos)):
            if 'small' in self.model_path:
                if (-2.00 <= pos[i][0] <= 2.00) and (-2.00 <= pos[i][1] <= 2.00):
                    bad_pos_idx.append(i)
                elif (2.75 <= pos[i][0]) or (pos[i][0] <= -2.75):
                    bad_pos_idx.append(i)
                elif (2.75 <= pos[i][1]) or (pos[i][1] <= -2.75):
                    bad_pos_idx.append(i)
            else:
                raise NotImplementedError

        return bad_pos_idx

    def _sample_uniform_xy(self, batch_size):
        goals = np.random.uniform(
            self.goal_space.low[:2],
            self.goal_space.high[:2],
            size=(batch_size, 2),
        )

        bad_goals_idx = self._collision_idx(goals)
        goals = np.delete(goals, bad_goals_idx, axis=0)
        while len(bad_goals_idx) > 0:
            new_goals = np.random.uniform(
                self.goal_space.low[:2],
                self.goal_space.high[:2],
                size=(len(bad_goals_idx), 2),
            )

            bad_goals_idx = self._collision_idx(new_goals)
            new_goals = np.delete(new_goals, bad_goals_idx, axis=0)
            goals = np.concatenate((goals, new_goals))

        # if 'small' in self.model_path:
        #     goals[(0 <= goals) * (goals < 0.5)] += 1
        #     goals[(0 <= goals) * (goals < 1.25)] += 1
        #     goals[(0 >= goals) * (goals > -0.5)] -= 1
        #     goals[(0 >= goals) * (goals > -1.25)] -= 1
        # else:
        #     goals[(0 <= goals) * (goals < 0.5)] += 2
        #     goals[(0 <= goals) * (goals < 1.5)] += 1.5
        #     goals[(0 >= goals) * (goals > -0.5)] -= 2
        #     goals[(0 >= goals) * (goals > -1.5)] -= 1.5
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
    env = gym.make('AntMaze150RandomInitEnv-v0')
    # env = gym.make('AntCrossMaze150Env-v0')
    # env = gym.make('DebugAntMaze30BottomLeftRandomInitGoalsPreset1Env-v0')
    env = gym.make(
        # 'AntMaze30RandomInitFS20Env-v0',
        # 'AntMaze30RandomInitEnv-v0',
        # 'AntMazeSmall30RandomInitFS10Env-v0',
        # 'AntMazeSmall30RandomInitFs5Dt3Env-v0',
        # 'AntMaze30RandomInitNoVelEnv-v0',
        # 'AntMaze30StateEnv-v0',
        # 'AntMaze30QposRandomInitFS20Env-v0',
        # 'AntMazeSmall30RandomInitFs10Dt3Env-v0',
        # 'AntMazeQposRewSmall30RandomInitFs5Dt3Env-v0',
        # 'AntMazeXyRewSmall30RandomInitFs5Dt3Env-v0',
        # 'AntMazeQposRewSmall30Fs5Dt3Env-v0',
        # 'AntMazeQposRewSmall30Fs5Dt3NoTermEnv-v0',
        # 'AntMazeXyRewSmall30RandomInitFs5Dt3NoTermEnv-v0',
        # 'AntMazeXyRewSmall30Fs5Dt3NoTermEnv-v0',
        'AntMazeQposRewSmall30Fs5Dt3NoTermEnv-v0',
    )
    env.reset()
    i = 0
    while True:
        i += 1
        env.render()
        action = env.action_space.sample()
        # action = np.zeros_like(action)
        obs, reward, done, info = env.step(action)
        # print(reward, np.linalg.norm(env.sim.data.get_body_xpos('torso')[:2]
        #                              - env._xy_goal) )
        # print(env.sim.data.qpos)
        print(info)
        if i % 5 == 0:
            env.reset()
