import numpy as np

from multiworld.envs.mujoco.classic_mujoco.ant import AntEnv


class AntMazeEnv(AntEnv):

    def __init__(
            self,
            # model_path='classic_mujoco/normal_gear_ratio_ant.xml',
            # test_mode_case_num=None,
            *args,
            **kwargs
    ):
        self.ant_radius = 0.75

        wall_collision_buffer = kwargs.get("wall_collision_buffer", 0.0)
        self.wall_radius = self.ant_radius + wall_collision_buffer

        model_path = kwargs['model_path']
        test_mode_case_num = kwargs.get('test_mode_case_num', None)

        if model_path in [
            'classic_mujoco/ant_maze2_gear30_small_dt3.xml',
            'classic_mujoco/ant_gear30_dt3_u_small.xml',
        ]:
            self.maze_type = 'u-small'
        elif model_path in [
            'classic_mujoco/ant_gear30_dt3_u_med.xml',
            'classic_mujoco/ant_gear15_dt3_u_med.xml',
            'classic_mujoco/ant_gear10_dt3_u_med.xml',
            'classic_mujoco/ant_gear30_dt2_u_med.xml',
            'classic_mujoco/ant_gear15_dt2_u_med.xml',
            'classic_mujoco/ant_gear10_dt2_u_med.xml',
            'classic_mujoco/ant_gear30_u_med.xml',
        ]:
            self.maze_type = 'u-med'
        elif model_path in [
            'classic_mujoco/ant_maze2_gear30_big_dt3.xml',
            'classic_mujoco/ant_gear30_dt3_u_big.xml',
        ]:
            self.maze_type = 'u-big'
        elif model_path in [
            'classic_mujoco/ant_gear10_dt3_u_long.xml',
            'classic_mujoco/ant_gear15_dt3_u_long.xml',
        ]:
            self.maze_type = 'u-long'
        elif model_path in [
            'classic_mujoco/ant_gear10_dt3_no_walls_long.xml',
            'classic_mujoco/ant_gear15_dt3_no_walls_long.xml',
        ]:
            self.maze_type = 'no-walls-long'
        elif model_path == 'classic_mujoco/ant_fb_gear30_small_dt3.xml':
            self.maze_type = 'fb-small'
        elif model_path in [
            'classic_mujoco/ant_fb_gear30_med_dt3.xml',
            'classic_mujoco/ant_gear30_dt3_fb_med.xml',
            'classic_mujoco/ant_gear15_dt3_fb_med.xml',
            'classic_mujoco/ant_gear10_dt3_fb_med.xml',
        ]:
            self.maze_type = 'fb-med'
        elif model_path == 'classic_mujoco/ant_fb_gear30_big_dt3.xml':
            self.maze_type = 'fb-big'
        elif model_path == 'classic_mujoco/ant_fork_gear30_med_dt3.xml':
            self.maze_type = 'fork-med'
        elif model_path == 'classic_mujoco/ant_fork_gear30_big_dt3.xml':
            self.maze_type = 'fork-big'
        elif model_path in [
            'classic_mujoco/ant_gear10_dt3_maze_med.xml',
        ]:
            self.maze_type = 'maze-med'
        elif model_path in [
            'classic_mujoco/ant_gear10_dt3_fg_med.xml',
        ]:
            self.maze_type = 'fg-med'
        else:
            raise NotImplementedError

        if self.maze_type == 'u-small':
            self.walls = [
                Wall(0, 1.125, 1.25, 2.375, self.wall_radius),

                Wall(0, 4.5, 3.5, 1, self.wall_radius),
                Wall(0, -4.5, 3.5, 1, self.wall_radius),
                Wall(4.5, 0, 1, 5.5, self.wall_radius),
                Wall(-4.5, 0, 1, 5.5, self.wall_radius),
            ]

            if test_mode_case_num == 1:
                kwargs['reset_low'] = np.array([-2.5, 2.5])
                kwargs['reset_high'] = np.array([-2.25, 2.5])

                kwargs['goal_low'] = np.array([2.25, 2.5])
                kwargs['goal_high'] = np.array([2.5, 2.5])

        elif self.maze_type == 'u-med':
            self.walls = [
                Wall(0, 1.5, 1.5, 3, self.wall_radius),

                Wall(0, 5.5, 4.5, 1, self.wall_radius),
                Wall(0, -5.5, 4.5, 1, self.wall_radius),
                Wall(5.5, 0, 1, 6.5, self.wall_radius),
                Wall(-5.5, 0, 1, 6.5, self.wall_radius),
            ]

            if test_mode_case_num == 1:
                kwargs['reset_low'] = np.array([-3.25, 2.75])
                kwargs['reset_high'] = np.array([-2.75, 3.25])

                kwargs['goal_low'] = np.array([2.75, 2.75])
                kwargs['goal_high'] = np.array([3.25, 3.25])

            if 'goal_low' not in kwargs:
                kwargs['goal_low'] = np.array([-5.5, -5.5])
            if 'goal_high' not in kwargs:
                kwargs['goal_high'] = np.array([5.5, 5.5])

        elif self.maze_type == 'u-long':
            self.walls = [
                Wall(0, 1.5, 0.75, 7.5, self.wall_radius),

                Wall(0, 10.0, 3.75, 1, self.wall_radius),
                Wall(0, -10.0, 3.75, 1, self.wall_radius),
                Wall(4.75, 0, 1, 11.0, self.wall_radius),
                Wall(-4.75, 0, 1, 11.0, self.wall_radius),
            ]

            if test_mode_case_num == 1:
                kwargs['reset_low'] = np.array([-2.5, 7.25])
                kwargs['reset_high'] = np.array([-2.0, 7.75])

                kwargs['goal_low'] = np.array([2.0, 7.25])
                kwargs['goal_high'] = np.array([2.5, 7.75])

            if 'goal_low' not in kwargs:
                kwargs['goal_low'] = np.array([-3.75, -9.0])
            if 'goal_high' not in kwargs:
                kwargs['goal_high'] = np.array([3.75, 9.0])

        elif self.maze_type == 'no-walls-long':
            self.walls = [
                Wall(0, 10.0, 3.75, 1, self.wall_radius),
                Wall(0, -10.0, 3.75, 1, self.wall_radius),
                Wall(4.75, 0, 1, 11.0, self.wall_radius),
                Wall(-4.75, 0, 1, 11.0, self.wall_radius),
            ]

            if test_mode_case_num == 1:
                kwargs['reset_low'] = np.array([-2.5, -7.75])
                kwargs['reset_high'] = np.array([-2.0, -7.25])

                kwargs['goal_low'] = np.array([2.0, 7.25])
                kwargs['goal_high'] = np.array([2.5, 7.75])

            if 'goal_low' not in kwargs:
                kwargs['goal_low'] = np.array([-3.75, -9.0])
            if 'goal_high' not in kwargs:
                kwargs['goal_high'] = np.array([3.75, 9.0])

        elif self.maze_type == 'fb-small':
            self.walls = [
                Wall(-2.0, 1.25, 0.75, 4.0, self.wall_radius),
                Wall(2.0, -1.25, 0.75, 4.0, self.wall_radius),

                Wall(0, 6.25, 5.25, 1, self.wall_radius),
                Wall(0, -6.25, 5.25, 1, self.wall_radius),
                Wall(6.25, 0, 1, 7.25, self.wall_radius),
                Wall(-6.25, 0, 1, 7.25, self.wall_radius),
            ]

        elif self.maze_type == 'fb-med':
            self.walls = [
                Wall(-2.25, 1.5, 0.75, 4.5, self.wall_radius),
                Wall(2.25, -1.5, 0.75, 4.5, self.wall_radius),

                Wall(0, 7.0, 6.0, 1, self.wall_radius),
                Wall(0, -7.0, 6.0, 1, self.wall_radius),
                Wall(7.0, 0, 1, 8.0, self.wall_radius),
                Wall(-7.0, 0, 1, 8.0, self.wall_radius),
            ]

            if test_mode_case_num == 1:
                kwargs['reset_low'] = np.array([-4.75, 4.25])
                kwargs['reset_high'] = np.array([-4.25, 4.75])

                kwargs['goal_low'] = np.array([4.25, -4.75])
                kwargs['goal_high'] = np.array([4.75, -4.25])
            elif test_mode_case_num == 2:
                kwargs['reset_low'] = np.array([-4.75, -0.25])
                kwargs['reset_high'] = np.array([-4.25, 0.25])

                kwargs['goal_low'] = np.array([4.25, -0.25])
                kwargs['goal_high'] = np.array([4.75, -0.25])
            elif test_mode_case_num == 3:
                kwargs['reset_low'] = np.array([-4.75, -4.75])
                kwargs['reset_high'] = np.array([-4.25, -4.25])

                kwargs['goal_low'] = np.array([4.25, 4.25])
                kwargs['goal_high'] = np.array([4.75, 4.75])
            elif test_mode_case_num == 4:
                kwargs['reset_low'] = np.array([-4.75, 4.25])
                kwargs['reset_high'] = np.array([-4.25, 4.75])

                kwargs['goal_low'] = np.array([-0.25, 4.25])
                kwargs['goal_high'] = np.array([0.25, 4.75])

            if 'goal_low' not in kwargs:
                kwargs['goal_low'] = np.array([-7.0, -7.0])
            if 'goal_high' not in kwargs:
                kwargs['goal_high'] = np.array([7.0, 7.0])

        elif self.maze_type == 'fb-big':
            self.walls = [
                Wall(-2.75, 2.0, 0.75, 5.5, self.wall_radius),
                Wall(2.75, -2.0, 0.75, 5.5, self.wall_radius),

                Wall(0, 8.5, 7.5, 1, self.wall_radius),
                Wall(0, -8.5, 7.5, 1, self.wall_radius),
                Wall(8.5, 0, 1, 9.5, self.wall_radius),
                Wall(-8.5, 0, 1, 9.5, self.wall_radius),
            ]
        elif self.maze_type == 'fork-med':
            self.walls = [
                Wall(-1.75, -1.5, 0.25, 3.5, self.wall_radius),
                Wall(0, 1.75, 2.0, 0.25, self.wall_radius),
                Wall(0, -1.75, 2.0, 0.25, self.wall_radius),

                Wall(0, 6.0, 5.0, 1, self.wall_radius),
                Wall(0, -6.0, 5.0, 1, self.wall_radius),
                Wall(6.0, 0, 1, 7.0, self.wall_radius),
                Wall(-6.0, 0, 1, 7.0, self.wall_radius),
            ]

            if test_mode_case_num == 1:
                kwargs['reset_low'] = np.array([-0.25, -3.75])
                kwargs['reset_high'] = np.array([0.25, -3.25])

                kwargs['goal_low'] = np.array([-3.75, -3.75])
                kwargs['goal_high'] = np.array([-3.25, -3.25])
            elif test_mode_case_num == 2:
                kwargs['reset_low'] = np.array([-0.25, -0.25])
                kwargs['reset_high'] = np.array([0.25, 0.25])

                kwargs['goal_low'] = np.array([-3.75, -0.25])
                kwargs['goal_high'] = np.array([-3.25, 0.25])

        elif self.maze_type == 'fork-big':
            self.walls = [
                Wall(-3.5, -1.5, 0.25, 5.25, self.wall_radius),
                Wall(0, -3.5, 3.75, 0.25, self.wall_radius),
                Wall(0, 0.0, 3.75, 0.25, self.wall_radius),
                Wall(0, 3.5, 3.75, 0.25, self.wall_radius),

                Wall(0, 7.75, 6.75, 1, self.wall_radius),
                Wall(0, -7.75, 6.75, 1, self.wall_radius),
                Wall(7.75, 0, 1, 8.75, self.wall_radius),
                Wall(-7.75, 0, 1, 8.75, self.wall_radius),
            ]

            if test_mode_case_num == 1:
                kwargs['reset_low'] = np.array([-5.5, -5.5])
                kwargs['reset_high'] = np.array([-5.0, -5.0])

                kwargs['goal_low'] = np.array([-5.5, -5.5])
                kwargs['goal_high'] = np.array([-5.0, -5.0])

        elif self.maze_type == 'maze-med':
            self.walls = [
                Wall(2.375, 3.25, 1.625, 0.75, self.wall_radius),
                Wall(-2.375, 3.25, 1.625, 0.75, self.wall_radius),
                Wall(0, 2, 0.75, 6, self.wall_radius),
                Wall(6, -2.25, 2, 0.75, self.wall_radius),
                Wall(-6, -2.25, 2, 0.75, self.wall_radius),

                Wall(0, 9, 8, 1, self.wall_radius),
                Wall(0, -9, 8, 1, self.wall_radius),
                Wall(9, 0, 1, 10, self.wall_radius),
                Wall(-9, 0, 1, 10, self.wall_radius),
            ]

            if test_mode_case_num == 1:
                kwargs['reset_low'] = np.array([-2.5, 6.25])
                kwargs['reset_high'] = np.array([-2.0, 6.75])

                kwargs['goal_low'] = np.array([2.0, 6.25])
                kwargs['goal_high'] = np.array([2.5, 6.75])
            elif test_mode_case_num == 2:
                kwargs['reset_low'] = np.array([0.75, 6.25])
                kwargs['reset_high'] = np.array([1.25, 6.75])

                kwargs['goal_low'] = np.array([0.75, 6.25])
                kwargs['goal_high'] = np.array([1.25, 6.75])

            if 'goal_low' not in kwargs:
                kwargs['goal_low'] = np.array([-8.0, -8.0])
            if 'goal_high' not in kwargs:
                kwargs['goal_high'] = np.array([8.0, 8.0])

        elif self.maze_type == 'fg-med':
            self.walls = [
                Wall(0, -3.25, 3, 0.75, self.wall_radius),
                Wall(-3.75, 0, 0.75, 4, self.wall_radius),
                Wall(3.75, 0, 0.75, 4, self.wall_radius),
                Wall(0, -6, 0.75, 2, self.wall_radius),

                Wall(0, 9, 8, 1, self.wall_radius),
                Wall(0, -9, 8, 1, self.wall_radius),
                Wall(9, 0, 1, 10, self.wall_radius),
                Wall(-9, 0, 1, 10, self.wall_radius),
            ]

            if test_mode_case_num == 1:
                kwargs['reset_low'] = np.array([-2.5, -6.75])
                kwargs['reset_high'] = np.array([-2.0, -6.25])

                kwargs['goal_low'] = np.array([2.0, -6.75])
                kwargs['goal_high'] = np.array([2.5, -6.25])
            elif test_mode_case_num == 2:
                kwargs['reset_low'] = np.array([-0.25, -1.25])
                kwargs['reset_high'] = np.array([0.25, -0.75])

                kwargs['goal_low'] = np.array([2.0, -6.75])
                kwargs['goal_high'] = np.array([2.5, -6.25])
            elif test_mode_case_num == 3:
                kwargs['reset_low'] = np.array([-6.25, -0.25])
                kwargs['reset_high'] = np.array([-5.75, 0.25])

                kwargs['goal_low'] = np.array([5.75, -0.25])
                kwargs['goal_high'] = np.array([6.25, 0.25])

            if 'goal_low' not in kwargs:
                kwargs['goal_low'] = np.array([-8.0, -8.0])
            if 'goal_high' not in kwargs:
                kwargs['goal_high'] = np.array([8.0, 8.0])
        else:
            raise NotImplementedError

        self.quick_init(locals())
        AntEnv.__init__(
            self,
            # model_path=model_path,
            *args,
            **kwargs
        )


    def _collision_idx(self, pos):
        bad_pos_idx = []
        for i in range(len(pos)):
            for wall in self.walls:
                if wall.contains_point(pos[i]):
                    bad_pos_idx.append(i)
                    break

            # if 'small' in self.model_path:
            #     if 'maze2' in self.model_path:
            #         if (-2.00 <= pos[i][0] <= 2.00) and (-2.00 <= pos[i][1]):
            #             bad_pos_idx.append(i)
            #         elif (2.75 <= pos[i][0]) or (pos[i][0] <= -2.75):
            #             bad_pos_idx.append(i)
            #         elif (2.75 <= pos[i][1]) or (pos[i][1] <= -2.75):
            #             bad_pos_idx.append(i)
            #     else:
            #         if (-2.00 <= pos[i][0] <= 2.00) and (-2.00 <= pos[i][1] <= 2.00):
            #             bad_pos_idx.append(i)
            #         elif (2.75 <= pos[i][0]) or (pos[i][0] <= -2.75):
            #             bad_pos_idx.append(i)
            #         elif (2.75 <= pos[i][1]) or (pos[i][1] <= -2.75):
            #             bad_pos_idx.append(i)
            # elif 'big' in self.model_path:
            #     if 'maze2' in self.model_path:
            #         if (-2.25 <= pos[i][0] <= 2.25) and (-2.75 <= pos[i][1]):
            #             bad_pos_idx.append(i)
            #         elif (4.75 <= pos[i][0]) or (pos[i][0] <= -4.75):
            #             bad_pos_idx.append(i)
            #         elif (4.75 <= pos[i][1]) or (pos[i][1] <= -4.75):
            #             bad_pos_idx.append(i)
            #     else:
            #         raise NotImplementedError
            # else:
            #     raise NotImplementedError

        return bad_pos_idx

    def _sample_uniform_xy(self, batch_size, mode='goal'):
        assert mode in ['reset', 'goal']

        if mode == 'reset':
            low, high = self.reset_low, self.reset_high
        elif mode == 'goal':
            low, high = self.goal_low, self.goal_high

        goals = np.random.uniform(
            low,
            high,
            size=(batch_size, 2),
        )

        bad_goals_idx = self._collision_idx(goals)
        goals = np.delete(goals, bad_goals_idx, axis=0)
        while len(bad_goals_idx) > 0:
            new_goals = np.random.uniform(
                low,
                high,
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

class Wall:
    def __init__(self, x_center, y_center, x_thickness, y_thickness, min_dist):
        self.min_x = x_center - x_thickness - min_dist
        self.max_x = x_center + x_thickness + min_dist
        self.min_y = y_center - y_thickness - min_dist
        self.max_y = y_center + y_thickness + min_dist

        self.endpoint1 = (x_center+x_thickness, y_center+y_thickness)
        self.endpoint2 = (x_center+x_thickness, y_center-y_thickness)
        self.endpoint3 = (x_center-x_thickness, y_center-y_thickness)
        self.endpoint4 = (x_center-x_thickness, y_center+y_thickness)

    def contains_point(self, point):
        return (self.min_x < point[0] < self.max_x) and (self.min_y < point[1] < self.max_y)

    def contains_points(self, points):
        return (self.min_x < points[:,0]) * (points[:,0] < self.max_x) \
               * (self.min_y < points[:,1]) * (points[:,1] < self.max_y)


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
