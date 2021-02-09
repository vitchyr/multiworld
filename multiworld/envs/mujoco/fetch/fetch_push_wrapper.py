import os
from gym import utils
from gym.envs.robotics import FetchPushEnv, fetch_env
import numpy as np

MODEL_XML_PATH = os.path.join('fetch', 'push.xml')


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class FetchPushCustomGoalSamplingEnv(FetchPushEnv):
    def __init__(
            self,
            fixed_goal_relative_xy=None,
            fixed_obj_relative_xy=None,
            reward_type='sparse',
            fix_init_position=True,
    ):
        self._fixed_goal_relative_xy = fixed_goal_relative_xy
        self._fixed_obj_relative_xy = fixed_obj_relative_xy
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
        }
        fetch_env.FetchEnv.__init__(
            self,
            MODEL_XML_PATH,
            has_object=True,
            block_gripper=True,
            n_substeps=20,
            gripper_extra_height=0.0,
            target_in_the_air=False,
            target_offset=0.0,
            obj_range=0.15,
            target_range=0.15,
            distance_threshold=0.05,
            initial_qpos=initial_qpos,
            reward_type=reward_type,
        )
        utils.EzPickle.__init__(self)
        self.fix_init_position = fix_init_position

    def _reset_sim(self):
        if (
                self._fixed_obj_relative_xy is None
                or not self.fix_init_position
        ):
            return super(FetchPushCustomGoalSamplingEnv, self)._reset_sim()

        self.sim.set_state(self.initial_state)

        # Randomize start position of object.
        if self.has_object:
            object_xpos = self.initial_gripper_xpos[:2]
            object_xpos = object_xpos + np.array(self._fixed_obj_relative_xy)
            object_qpos = self.sim.data.get_joint_qpos('object0:joint')
            assert object_qpos.shape == (7,)
            object_qpos[:2] = object_xpos
            self.sim.data.set_joint_qpos('object0:joint', object_qpos)

        self.sim.forward()
        return True

    def step(self, action):
        obs, reward, done, info = super().step(action)
        info['distance'] = goal_distance(obs['achieved_goal'], self.goal)
        return obs, reward, done, info

    def _sample_goal(self):
        if self._fixed_goal_relative_xy is None:
            return super()._sample_goal()
        else:
            return np.array([
                self.initial_gripper_xpos[0] + self._fixed_goal_relative_xy[0],
                self.initial_gripper_xpos[1] + self._fixed_goal_relative_xy[1],
                self.height_offset,
            ])


if __name__ == '__main__':
    env = FetchPushCustomGoalSamplingEnv(
        reward_type='sparse',
        fixed_goal_relative_xy=(.15, .15),
    )
    import gym
    from multiworld.envs.mujoco import register_extra_fetch_envs
    register_extra_fetch_envs()
    # env = gym.make('FetchPush-FixedGoal-x0p15-y0-v1')
    # env = gym.make('FetchPush-FixedGoal-x0p15-y0p15-v1')
    env = gym.make('FetchPush-FixedInit-RandomGoal-v1')
    for _ in range(1000):
        env.reset()
        for _ in range(10):
            env.step(env.action_space.sample())
            env.render(mode='human')
