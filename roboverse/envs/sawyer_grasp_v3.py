import gym
import numpy as np

from roboverse.envs.sawyer_grasp_v2 import SawyerGraspV2Env, REWARD_NEGATIVE, \
    load_shapenet_object
import roboverse.bullet as bullet


class SawyerGraspV3Env(SawyerGraspV2Env):

    def _set_action_space(self):
        act_dim = 5
        act_bound = 1
        act_high = np.ones(act_dim) * act_bound
        self.action_space = gym.spaces.Box(-act_high, act_high)

    def step(self, action):
        action = np.asarray(action)
        pos = list(bullet.get_link_state(self._sawyer, self._end_effector, 'pos'))
        delta_pos = action[:3]
        pos += delta_pos * self._action_scale
        pos = np.clip(pos, self._pos_low, self._pos_high)

        theta = list(bullet.get_link_state(self._sawyer, self._end_effector, 'theta'))
        delta_theta = action[3]
        target_theta = theta + np.asarray([0., 0., delta_theta*20])
        target_theta = np.clip(target_theta, [180, 0., 0.], [180, 0., 180.])
        target_theta = bullet.deg_to_quat(target_theta)
        gripper = -0.8

        self._simulate(pos, target_theta, gripper)
        # if self._visualize: self.visualize_targets(pos)

        pos = bullet.get_link_state(self._sawyer, self._end_effector, 'pos')
        if pos[2] < self._height_threshold and action[4] > 0.0:
            gripper = 0.8
            for i in range(10):
                self._simulate(pos, target_theta, gripper)
            for i in range(50):
                pos = bullet.get_link_state(self._sawyer, self._end_effector,
                                            'pos')
                pos = list(pos)
                pos = np.clip(pos, self._pos_low, self._pos_high)
                pos[2] += 0.05
                self._simulate(pos, target_theta, gripper)
            done = True
            reward = self.get_reward({})
            if reward > 0:
                info = {'grasp_success': 1.0}
            else:
                info = {'grasp_success': 0.0}
        else:
            done = False
            reward = REWARD_NEGATIVE
            info = {'grasp_success': 0.0}

        observation = self.get_observation()
        self._prev_pos = bullet.get_link_state(self._sawyer, self._end_effector,
                                               'pos')
        return observation, reward, done, info


class SawyerGraspV3ObjectSetEnv(SawyerGraspV3Env):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self._observation_mode == 'pixels'

    def _load_meshes(self):
        if self._invisible_robot:
            self._sawyer = bullet.objects.sawyer_invisible()
        else:
            self._sawyer = bullet.objects.sawyer_finger_visual_only()
        self._table = bullet.objects.table()
        self._tray = bullet.objects.tray()

        self._objects = {}
        self._sensors = {}
        self._workspace = bullet.Sensor(self._sawyer,
            xyz_min=self._pos_low, xyz_max=self._pos_high,
            visualize=False, rgba=[0,1,0,.1])


        self._end_effector = bullet.get_index_by_attribute(
            self._sawyer, 'link_name', 'gripper_site')

        import scipy.spatial
        min_distance_threshold = 0.12
        object_positions = np.random.uniform(
            low=self._object_position_low, high=self._object_position_high)
        object_positions = np.reshape(object_positions, (1, 3))
        while object_positions.shape[0] < self._num_objects:
            object_position_candidate = np.random.uniform(
                low=self._object_position_low, high=self._object_position_high)
            object_position_candidate = np.reshape(
                object_position_candidate, (1, 3))
            min_distance = scipy.spatial.distance.cdist(
                object_position_candidate, object_positions)
            if (min_distance > min_distance_threshold).any():
                object_positions = np.concatenate(
                    (object_positions, object_position_candidate), axis=0)

        assert len(self.object_ids) >= self._num_objects
        import random
        indexes = list(range(len(self.object_ids)))
        random.shuffle(indexes)
        indexes = indexes[:self._num_objects]
        for i, idx in enumerate(indexes):
            key_idx = self.object_ids.index(self.object_ids[idx])
            self._objects[key_idx] = load_shapenet_object(
                self.object_list[self.object_ids[idx]], self.scaling,
                object_positions[i], scale_local=self._scaling_local[idx])
            for _ in range(10):
                bullet.step()


if __name__ == "__main__":
    import roboverse
    env = roboverse.make('SawyerGraspOneObjectSetTenV3-v0', gui=True,
                         observation_mode='pixels')
    for _ in range(10):
        env.reset()
        for _ in range(50):
            env.step(env.action_space.sample())

