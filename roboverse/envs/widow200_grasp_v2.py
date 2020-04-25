import roboverse.bullet as bullet
import numpy as np
from roboverse.envs.sawyer_base import SawyerBaseEnv
from roboverse.envs.widowx200_grasp import WidowX200GraspEnv
import gym
from roboverse.bullet.misc import load_obj
import os.path as osp
import pickle

REWARD_NEGATIVE = -1.0
REWARD_POSITIVE = 10.0
SHAPENET_ASSET_PATH = osp.join(
    osp.dirname(osp.abspath(__file__)), 'assets/ShapeNetCore')


def load_shapenet_object(object_path, scaling, object_position, scale_local=0.5):
    path = object_path.split('/')
    dir_name = path[-2]
    object_name = path[-1]
    obj = load_obj(
        SHAPENET_ASSET_PATH + '/ShapeNetCore_vhacd/{0}/{1}/model.obj'.format(
            dir_name, object_name),
        SHAPENET_ASSET_PATH + '/ShapeNetCore.v2/{0}/{1}/models/model_normalized.obj'.format(
            dir_name, object_name),
        object_position,
        [0, 0, 1, 0],
        scale=scale_local*scaling[
            '{0}/{1}'.format(dir_name, object_name)])
    return obj

class Widow200GraspV2Env(WidowX200GraspEnv):
    def __init__(self, *args, **kwargs):
        self._object_position_high = (.83, .05, -.20)
        self._object_position_low = (.77, -.15, -.20)
        self._num_objects = 1
        self.object_ids = [[0, 1, 25, 30, 50, 215, 255, 265, 300, 310][1]]
        shapenet_data = pickle.load(
            open(osp.join(SHAPENET_ASSET_PATH, 'metadata.pkl'), 'rb'))
        self.object_list = shapenet_data['object_list']
        self.scaling = shapenet_data['scaling']
        self._scaling_local = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.3, 0.5, 0.5, 0.5]
        super().__init__(*args, **kwargs)
        self._env_name = 'Widow200GraspV2Env'
        self._height_threshold = -0.31
        
    def _load_meshes(self):
        super()._load_meshes()
        self._tray = bullet.objects.widow200_tray()

        self._objects = {}
        self._sensors = {}
        self._workspace = bullet.Sensor(self._robot_id,
            xyz_min=self._pos_low, xyz_max=self._pos_high,
            visualize=False, rgba=[0,1,0,.1])

        # TODO(avi) Add more objects
        import scipy.spatial
        min_distance_threshold = 0.12
        object_positions = np.random.uniform(
            low=self._object_position_low, high=self._object_position_high)
        object_positions = np.reshape(object_positions, (1,3))
        while object_positions.shape[0] < self._num_objects:
            object_position_candidate = np.random.uniform(
                low=self._object_position_low, high=self._object_position_high)
            object_position_candidate = np.reshape(
                object_position_candidate, (1,3))
            min_distance = scipy.spatial.distance.cdist(
                object_position_candidate, object_positions)
            if (min_distance > min_distance_threshold).any():
                object_positions = np.concatenate(
                    (object_positions, object_position_candidate), axis=0)

        assert len(self.object_ids) >= self._num_objects
        import random
        indexes = list(range(self._num_objects))
        random.shuffle(indexes)
        for idx in indexes:
            key_idx = self.object_ids.index(self.object_ids[idx])
            self._objects[key_idx] = load_shapenet_object(
                self.object_list[self.object_ids[idx]], self.scaling,
                object_positions[idx], scale_local=self._scaling_local[idx])
            for _ in range(10):
                bullet.step()

    def step(self, *action):
        delta_pos, gripper = self._format_action(*action)
        pos = bullet.get_link_state(self._robot_id, self._end_effector, 'pos')
        pos += delta_pos * self._action_scale
        pos = np.clip(pos, self._pos_low, self._pos_high)

        if len(action) > 3:
            theta = list(bullet.get_link_state(self._robot_id, self._end_effector, 'theta'))
            delta_theta = action[3]
            target_theta = theta + np.asarray([0., 0., delta_theta*20])
            target_theta = np.clip(target_theta, [180, 0., 0.], [180, 0., 180.])
            target_theta = bullet.deg_to_quat(target_theta)
            gripper = -0.8
        else:
            target_theta = self.theta

        self._simulate(pos, self.theta, gripper)
        # if self._visualize: self.visualize_targets(pos)

        pos = bullet.get_link_state(self._robot_id, self._end_effector, 'pos')
        if pos[2] < self._height_threshold:
            gripper = 0.8
            for i in range(10):
                self._simulate(pos, target_theta, gripper)
            for i in range(50):
                pos = bullet.get_link_state(self._robot_id, self._end_effector, 'pos')
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
        self._prev_pos = bullet.get_link_state(self._robot_id, self._end_effector, 'pos')
        return observation, reward, done, info

    def get_observation(self):
        left_tip_pos = bullet.get_link_state(
            self._robot_id, self._gripper_joint_name[0], keys='pos')
        right_tip_pos = bullet.get_link_state(
            self._robot_id, self._gripper_joint_name[1], keys='pos')
        left_tip_pos = np.asarray(left_tip_pos)
        right_tip_pos = np.asarray(right_tip_pos)

        gripper_tips_distance = [np.linalg.norm(
            left_tip_pos - right_tip_pos)]
        end_effector_pos = self.get_end_effector_pos()

        for object_name in range(self._num_objects):
            observation = np.concatenate(
                (end_effector_pos, gripper_tips_distance))

            object_info = bullet.get_body_info(self._objects[object_name],
                                                   quat_to_deg=False)
            object_pos = object_info['pos']
            object_theta = object_info['theta']
            observation = np.concatenate(
                (observation, object_pos, object_theta))

        return observation

    def get_info(self):
        return {}

    def get_reward(self, info):
        return 0

if __name__ == "__main__":
    import roboverse
    import time

    num_objects = 1
    # env = roboverse.make("SawyerGraspOneV2-v0",
    #                      gui=True,
    #                      observation_mode='state',)
    #                      # num_objects=num_objects)
    env = roboverse.make("Widow200GraspV2-v0", gui=True)
    obs = env.reset()
    # object_ind = np.random.randint(0, env._num_objects)
    object_ind = num_objects - 1
    i = 0
    action = env.action_space.sample()
    for _ in range(5000):
        time.sleep(0.1)
        object_pos = obs[4: 4 + 3]
        print("obs", obs)
        print("object_pos", object_pos)
        ee_pos = obs[:3]
        action = object_pos - ee_pos
        action = action*4.0
        action += np.random.normal(scale=0.1, size=(3,))

        # action = np.random.uniform(low=-1.0, high=1.0, size=(3,))
        # if np.random.uniform() < 0.9:
        #     action[2] = -1
        # theta_action = np.random.uniform()
        theta_action = 0.0

        action = np.concatenate((action, np.asarray([theta_action])))
        print('action', action)
        obs, rew, done, info = env.step(action)
        env.render()
        i+=1
        if done or i > 50:
            # object_ind = np.random.randint(0, env._num_objects)
            object_ind = num_objects - 1
            obs = env.reset()
            i = 0
            print('Reward: {}'.format(rew))
        # print(obs)

