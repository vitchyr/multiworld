import os
import time

import gym
import numpy as np
import pybullet as p
import pybullet_data
from gym import spaces

from roboverse.core.ik import sawyer_ik, position_control
from roboverse.core.misc import load_urdf, load_obj, load_random_objects
from roboverse.core.queries import get_index_by_attribute, get_link_state

LARGE_VAL_OBSERVATION = 100


class SawyerReachEnv(gym.Env):

    def __init__(self,
                 action_repeat=10,
                 renders=False,
                 goal_observation=(0.8, 0.4, 0.0),
                 control_xyz_position_only=True):

        self._time_step = 1. / 240.

        self._renders = renders
        self._action_repeat = action_repeat
        self._goal_observation = np.asarray(goal_observation)
        self._control_xyz_position_only = control_xyz_position_only
        self.terminated = 0

        self._p = p

        if self._renders:
            cid = p.connect(p.SHARED_MEMORY)
            if cid < 0:
                p.connect(p.GUI)
            p.resetDebugVisualizerCamera(1.3, 180, -41, [0.52, -0.2, -0.33])
        else:
            p.connect(p.DIRECT)

        self._set_paths()
        obs = self.reset()
        observation_dim = len(obs)

        observation_high = np.array([LARGE_VAL_OBSERVATION] * observation_dim)

        action_dim = 5
        self._action_bound = 1
        action_high = np.array([self._action_bound] * action_dim)
        self.action_space = spaces.Box(-action_high, action_high)
        self.observation_space = spaces.Box(-observation_high, observation_high)
        self.viewer = None

    def _set_paths(self):
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(curr_dir, 'assets/sawyer_robot')
        self._sawyer_urdf_path = os.path.join(
            model_dir, 'sawyer_description/urdf/sawyer_xacro.urdf')
        self._pybullet_data_dir = pybullet_data.getDataPath()
        self._object_path = 'assets/ShapeNetSem'

    def reset(self):
        p.resetSimulation()

        ## load meshes
        self._sawyer = load_urdf(self._sawyer_urdf_path)
        self._table = load_urdf(
            os.path.join(self._pybullet_data_dir, 'table/table.urdf'),
            [.75, -.2, -1], [0, 0, 0.707107, 0.707107],
            scale=1.0)
        self._duck = load_urdf(
            os.path.join(self._pybullet_data_dir, 'duck_vhacd.urdf'),
            [.75, -.2, 0], [0, 0, 1, 0], scale=0.8)
        self._lego = load_urdf(
            os.path.join(self._pybullet_data_dir, 'lego/lego.urdf'),
            [.75, .2, 0], [0, 0, 1, 0], rgba=[1, 0, 0, 1],
            scale=1.5)
        self._end_effector = get_index_by_attribute(
            self._sawyer, 'link_name', 'right_l6')
        load_random_objects(self._object_path, 3)

        p.setPhysicsEngineParameter(numSolverIterations=150)
        p.setTimeStep(self._time_step)
        p.setGravity(0, 0, -10)
        p.stepSimulation()
        pos = np.array([0.5, 0, 0])
        self.theta = [0.7071, 0.7071, 0, 0]
        position_control(self._sawyer, self._end_effector, pos, self.theta)
        return self.get_observation()

    def get_observation(self):
        observation = get_link_state(self._sawyer, self._end_effector, 'pos')
        return np.asarray(observation)

    def step(self, action, angle):
        pos = get_link_state(self._sawyer, self._end_effector, 'pos')
        pos += action[:3] * 0.1
        if not self._control_xyz_position_only:
            if not hasattr(self, 'theta'):
                self.theta = [0.7071, 0.7071, 0, 0]
            self.theta += angle[:4] * 0.1
        else:
            self.theta = [0.7071, 0.7071, 0, 0]
        gripper = 0
        done = False
        for _ in range(self._action_repeat):
            sawyer_ik(self._sawyer, self._end_effector, pos, self.theta, gripper)
            p.stepSimulation()
        observation = self.get_observation()
        return observation, self.get_reward(observation), done, {}

    def render(self, mode='human'):
        # TODO
        pass

    def get_reward(self, observation):
        return -1.0 * np.linalg.norm(observation - self._goal_observation)


if __name__ == "__main__":
    env = SawyerReachEnv(renders=True)
    env.reset()
    for _ in range(1000):
        time.sleep(0.1)
        act = np.array([1.0, 1.0, 0.0])
        print(act)
        print(env.step(act))

    # t0 = time.time()
    # env = SawyerReachEnv(renders=False)
    # act = np.array([1.0, 1.0, 0.0])
    # for _ in range(10):
    #     env.reset()
    #     for _ in range(100):
    #         env.step(act)
    # t1 = time.time()
    # total = t1 - t0
    # print(total)
