from roboverse.envs.sawyer_grasp_v4 import SawyerGraspV4Env
from roboverse.bullet.misc import load_obj
import roboverse.bullet as bullet
from gym.spaces import Box, Dict
import numpy as np
import gym

REWARD_NEGATIVE = 0.0
REWARD_POSITIVE = 1.0
SHAPENET_ASSET_PATH = "/home/ashvin/ros_ws/src/ashvindev/bullet-objects/ShapeNetCore/"

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

class SawyerRigGraspV1Env(SawyerGraspV4Env):
    
    def __init__(self,
                 observation_mode='state',
                 num_objects=1,
                 obs_img_dim=84,
                 transpose_image=False,
                 invisible_robot=False,
                 reward_type=False, # Not actually used
                 randomize=True, # Not actually used
                 height_threshold=-0.32,
                 reward_height_thresh=-0.3,
                 object_position_low =(.60, .05, -.20),
                 object_position_high =(.80, .25, -.20),
                 object_ids = [1],#[0, 1, 25, 30, 50, 215, 255, 265, 300, 310],
                 *args,
                 **kwargs
                 ):
        super().__init__(
            observation_mode,
            num_objects,
            obs_img_dim,
            transpose_image,
            invisible_robot,
            reward_type,
            randomize,
            height_threshold,
            reward_height_thresh,
            object_position_low, 
            object_position_high,
            object_ids,
            *args,
            **kwargs
            )

    def get_observation(self, just_pos=False):
        left_tip_pos = bullet.get_link_state(
            self._sawyer, 'right_gripper_l_finger_joint', keys='pos')
        right_tip_pos = bullet.get_link_state(
            self._sawyer, 'right_gripper_r_finger_joint', keys='pos')
        left_tip_pos = np.asarray(left_tip_pos)
        right_tip_pos = np.asarray(right_tip_pos)

        gripper_tips_distance = [np.linalg.norm(
            left_tip_pos - right_tip_pos)]
        end_effector_pos = self.get_end_effector_pos()
        end_effector_theta = bullet.get_link_state(
            self._sawyer, self._end_effector, 'theta', quat_to_deg=False)

        if self._observation_mode == 'state':
            observation = np.concatenate(
                (end_effector_pos, end_effector_theta, gripper_tips_distance))
            # object_list = self._objects.keys()
            for object_name in range(self._num_objects):
                object_info = bullet.get_body_info(self._objects[object_name],
                                                   quat_to_deg=False)
                object_pos = object_info['pos']
                object_theta = object_info['theta']

                observation = np.concatenate(
                    (observation, object_pos, object_theta))
        elif self._observation_mode == 'pixels':
            image_observation = self.render_obs()
            image_observation = np.float32(image_observation.flatten())/255.0
            # image_observation = np.zeros((48, 48, 3), dtype=np.uint8)
            observation = {
                'state': np.concatenate(
                    (end_effector_pos, gripper_tips_distance)),
                'image': image_observation
            }
        elif self._observation_mode == 'pixels_debug':
            # This mode passes in all the true state information + images
            image_observation = self.render_obs()
            image_observation = np.float32(image_observation.flatten())/255.0
            state_observation = np.concatenate(
                (end_effector_pos, end_effector_theta, gripper_tips_distance))

            for object_name in range(self._num_objects):
                object_info = bullet.get_body_info(self._objects[object_name],
                                                   quat_to_deg=False)
                object_pos = object_info['pos']
                object_theta = object_info['theta']
                state_observation = np.concatenate(
                    (state_observation, object_pos, object_theta))
            observation = {
                'state': state_observation,
                'image': image_observation,
            }
        else:
            raise NotImplementedError

        if just_pos:
            return observation

        observation = {
                'observation': observation,
                'state_observation': observation,
                'desired_goal': observation,#self.goal_state,
                'state_desired_goal': observation,#self.goal_state,
                'achieved_goal': observation,
                'state_achieved_goal': observation,
                'objects': observation}

        return observation

    def _set_spaces(self):

        self.objects_box = gym.spaces.Box(
            np.zeros((1 + self._num_objects, )),
            np.ones((1 + self._num_objects, )),
        )

        self._set_action_space()
        # obs = self.reset()
        if self._observation_mode == 'state':
            observation_dim = 7 + 1 + 7 * self._num_objects
            obs_bound = 100
            obs_high = np.ones(observation_dim) * obs_bound
            self.obs_box = gym.spaces.Box(-obs_high, obs_high)
        elif self._observation_mode == 'pixels' or self._observation_mode == 'pixels_debug':
            img_space = gym.spaces.Box(0, 1, (self.image_length,), dtype=np.float32)
            if self._observation_mode == 'pixels':
                observation_dim = 7
            elif self._observation_mode == 'pixels_debug':
                observation_dim = 7 + 1 + 7*self._num_objects
            obs_bound = 100
            obs_high = np.ones(observation_dim) * obs_bound
            state_space = gym.spaces.Box(-obs_high, obs_high)
            spaces = {'image': img_space, 'state': state_space}
            self.observation_space = gym.spaces.Dict(spaces)
        else:
            raise NotImplementedError
        
        self.observation_space = Dict([
            ('observation', self.obs_box),
            ('state_observation', self.obs_box),
            ('desired_goal', self.obs_box),
            ('state_desired_goal', self.obs_box),
            ('achieved_goal', self.obs_box),
            ('state_achieved_goal', self.obs_box),
            ('objects', self.objects_box),
        ])

    def get_reward(self, info):
        object_list = self._objects.keys()
        reward = REWARD_NEGATIVE
        for object_name in object_list:
            object_info = bullet.get_body_info(self._objects[object_name],
                                               quat_to_deg=False)
            object_pos = np.asarray(object_info['pos'])
            object_height = object_pos[2]
            if object_height > self._reward_height_thresh:
                end_effector_pos = np.asarray(self.get_end_effector_pos())
                object_gripper_distance = np.linalg.norm(
                    object_pos - end_effector_pos)
                if object_gripper_distance < 0.1:
                    reward = REWARD_POSITIVE
        return reward

    def _get_obs(self):
        return self.get_observation()

    def get_env_state(self):
        return self.get_observation()


    def get_contextual_diagnostics(self, paths, contexts):
        diagnostics = OrderedDict()
        state_key = "state_observation"
        goal_key = "state_desired_goal"
        values = []
        for i in range(len(paths)):
            state = paths[i]["observations"][-1][state_key]
            goal = contexts[i][goal_key]
            distance = np.linalg.norm(state - goal)
            values.append(distance)
        diagnostics_key = goal_key + "/final/distance"
        diagnostics.update(create_stats_ordered_dict(
            diagnostics_key,
            values,
        ))

        values = []
        for i in range(len(paths)):
            for j in range(len(paths[i]["observations"])):
                state = paths[i]["observations"][j][state_key]
                goal = contexts[i][goal_key]
                distance = np.linalg.norm(state - goal)
                values.append(distance)
        diagnostics_key = goal_key + "/distance"
        diagnostics.update(create_stats_ordered_dict(
            diagnostics_key,
            values,
        ))
        return diagnostics

    #Temporary
    def compute_rewards(self, actions, obs):
        return np.zeros(obs['state_desired_goal'].shape[0])


    # def _load_meshes(self, positions=None):
    #     if self._invisible_robot:
    #         self._sawyer = bullet.objects.sawyer_invisible()
    #     else:
    #         self._sawyer = bullet.objects.sawyer_finger_visual_only()
    #     self._table = bullet.objects.table()
    #     self._tray = bullet.objects.tray()

    #     self._objects = {}
    #     self._sensors = {}
    #     self._workspace = bullet.Sensor(self._sawyer,
    #         xyz_min=self._pos_low, xyz_max=self._pos_high,
    #         visualize=False, rgba=[0,1,0,.1])


    #     self._end_effector = bullet.get_index_by_attribute(
    #         self._sawyer, 'link_name', 'gripper_site')

    #     # TODO(avi) Add more objects
    #     import scipy.spatial
    #     min_distance_threshold = 0.12
    #     object_positions = np.random.uniform(
    #         low=self._object_position_low, high=self._object_position_high)
    #     object_positions = np.reshape(object_positions, (1,3))
    #     while object_positions.shape[0] < self._num_objects:
    #         object_position_candidate = np.random.uniform(
    #             low=self._object_position_low, high=self._object_position_high)
    #         object_position_candidate = np.reshape(
    #             object_position_candidate, (1,3))
    #         min_distance = scipy.spatial.distance.cdist(
    #             object_position_candidate, object_positions)
    #         if (min_distance > min_distance_threshold).any():
    #             object_positions = np.concatenate(
    #                 (object_positions, object_position_candidate), axis=0)

    #     assert len(self.object_ids) >= self._num_objects
    #     import random
    #     indexes = list(range(self._num_objects))
    #     random.shuffle(indexes)
    #     for idx in indexes:
    #         key_idx = self.object_ids.index(self.object_ids[idx])
    #         self._objects[key_idx] = load_shapenet_object(
    #             self.object_list[self.object_ids[idx]], self.scaling,
    #             object_positions[idx], scale_local=self._scaling_local[idx])
    #         for _ in range(10):
    #             bullet.step()


    # def set_to_goal(self, goal):
    #     #bullet.reset()
    #     #bullet.setup_headless(self._timestep, solver_iterations=self._solver_iterations)
    #     assert len(self.object_ids) >= self._num_objects
    #     import random

    #     # object_positions = goal['state_desired_goal'][8:].reshape(-1, 3)
    #     # indexes = list(range(self._num_objects))
    #     # random.shuffle(indexes)

    #     # bullet.position_control(self._sawyer, self._end_effector, object_positions[0]) #NOT MOVING GRIPPER. IS THIS A PROBLEM?
    #     # for idx in indexes:
    #     #     key_idx = self.object_ids.index(self.object_ids[idx])
            
    #     #     self._objects[key_idx] = load_shapenet_object(
    #     #         self.object_list[self.object_ids[idx]], self.scaling,
    #     #         object_positions[idx], scale_local=self._scaling_local[idx])
    #     #     for _ in range(10):
    #     #         bullet.step()

    #     hand = goal['state_desired_goal'][:8]
    #     object_positions = goal['state_desired_goal'][8:].reshape(self._num_objects, -1)
    #     indexes = list(range(self._num_objects))
    #     random.shuffle(indexes)

    #     bullet.position_control(self._sawyer, self._end_effector, hand[:3], self.theta) #NOT MOVING GRIPPER. IS THIS A PROBLEM?
    #     for idx in indexes:
    #         key_idx = self.object_ids.index(self.object_ids[idx])

    #         self._objects[key_idx] = load_shapenet_object(
    #             self.object_list[self.object_ids[idx]], self.scaling,
    #             object_positions[idx][:3], scale_local=self._scaling_local[idx])
    #         for _ in range(10):
    #             bullet.step()

    # def set_to_goal(self, goal):
    #     bullet.position_control(self._sawyer, self._end_effector, goal['state_desired_goal'][:3], self.theta)

    # def set_env_state(self, goal):
    #     self.set_to_goal(goal)
        #bullet.position_control(self._sawyer, self._end_effector, goal['state_desired_goal'][:3], self.theta)


    def get_goal(self):
        return {
            'desired_goal': self.goal_state,
            'state_desired_goal': self.goal_state,
        }

    def get_image(self, width, height):
        image = np.float32(self.render_obs())
        return image

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

        gripper_action = action[4]

        # is_gripper_open = self._is_gripper_open()
        is_gripper_open = self._gripper_open
        if gripper_action > 0.5 and is_gripper_open:
            # keep it open
            gripper = -0.8
            self._simulate(pos, target_theta, gripper)
        elif gripper_action > 0.5 and not is_gripper_open:
            # gripper is currently closed and we want to open it
            gripper = -0.8
            for i in range(10):
                self._simulate(pos, target_theta, gripper)
            self._gripper_open = True
        elif gripper_action < -0.5 and not is_gripper_open:
            # keep it closed
            gripper = 0.8
            self._simulate(pos, target_theta, gripper)
        elif gripper_action < -0.5 and is_gripper_open:
            # gripper is open and we want to close it
            gripper = +0.8
            for i in range(10):
                self._simulate(pos, target_theta, gripper)
            self._gripper_open = False
        elif gripper_action <= 0.5 and gripper_action >= -0.5:
            # maintain current status
            if is_gripper_open:
                gripper = -0.8
            else:
                gripper = 0.8
            self._simulate(pos, target_theta, gripper)
            pass
        else:
            raise NotImplementedError

        # if self._visualize: self.visualize_targets(pos)

        if action[5] > 0.5:
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

    def reset(self):
        bullet.reset()
        bullet.setup_headless(self._timestep, solver_iterations=self._solver_iterations)
        self._load_meshes()
        self._format_state_query()
        self._goal_pos = np.random.uniform(low=self._object_position_low, high=self._object_position_high)
        self._prev_pos = np.array(self._pos_init)
        bullet.position_control(self._sawyer, self._end_effector, self._prev_pos, self.theta)

        # start with an open gripper
        self._gripper_open = True
        gripper = -0.8
        for _ in range(3):
            self._simulate(self._prev_pos, self.theta, gripper)

        return self.get_observation()

    # def reset(self):
    #     self.goal_state = self.sample_rollout_goal()
    #     self.sample_rollout_goal()
    #     # bullet.reset()
    #     # bullet.setup_headless(self._timestep, solver_iterations=self._solver_iterations)
    #     # self._load_meshes()
    #     # self._format_state_query()
    #     self._prev_pos = np.array(self._pos_init)

    #     #self.goal_state = self.sample_rollout_goal()

    #     bullet.position_control(self._sawyer, self._end_effector, self._prev_pos, self.theta)

    #     #import ipdb; ipdb.set_trace()

    #     # start with an open gripper
    #     self._gripper_open = True
    #     gripper = -0.8
    #     for _ in range(3):
    #         self._simulate(self._prev_pos, self.theta, gripper)

    #     return self.get_observation()


    # def sample_rollout_goal(self):
    #     bullet.reset()
    #     bullet.setup_headless(self._timestep, solver_iterations=self._solver_iterations)
        
    #     self._load_meshes()
    #     self._format_state_query()
    #     hand_pos = np.random.uniform(low=self._object_position_low, high=self._object_position_high)
    #     bullet.position_control(self._sawyer, self._end_effector, hand_pos, self.theta)

    #     self._gripper_open = True
    #     gripper = -0.8
    #     for _ in range(3):
    #         self._simulate(hand_pos, self.theta, gripper)

    #     return self.get_observation(just_pos=True)

        # object_positions = np.reshape(object_positions, (1,3))

        # import scipy.spatial
        # min_distance_threshold = 0.12
        # object_positions = np.random.uniform(
        #     low=self._object_position_low, high=self._object_position_high)
        # object_positions = np.reshape(object_positions, (1,3))
        # while object_positions.shape[0] < self._num_objects + 1:
        #     object_position_candidate = np.random.uniform(
        #         low=self._object_position_low, high=self._object_position_high)
        #     object_position_candidate = np.reshape(
        #         object_position_candidate, (1,3))
        #     min_distance = scipy.spatial.distance.cdist(
        #         object_position_candidate, object_positions)
        #     if (min_distance > min_distance_threshold).any():
        #         object_positions = np.concatenate(
        #             (object_positions, object_position_candidate), axis=0)

        # return object_positions.flatten()



if __name__ == "__main__":
    import roboverse
    env = roboverse.make("SawyerRigGrasp-v0",
                         gui=True,
                         observation_mode='state',)

    object_ind = 0
    for _ in range(50):
        obs = env.reset()
        # object_pos[2] = -0.30

        dist_thresh = 0.04 + np.random.normal(scale=0.01)

        for _ in range(25):
            ee_pos = obs[:3]
            object_pos = obs[object_ind * 7 + 8: object_ind * 7 + 8 + 3]
            # object_pos += np.random.normal(scale=0.02, size=(3,))

            object_gripper_dist = np.linalg.norm(object_pos - ee_pos)
            theta_action = 0.
            # theta_action = np.random.uniform()
            # print(object_gripper_dist)
            if object_gripper_dist > dist_thresh and env._gripper_open:
                # print('approaching')
                action = (object_pos - ee_pos) * 4.0
                action = np.concatenate((action, np.asarray([theta_action,0.,0.])))
            elif env._gripper_open:
                # print('gripper closing')
                action = (object_pos - ee_pos) * 4.0
                action = np.concatenate(
                    (action, np.asarray([0., -0.7, 0.])))
            elif object_pos[2] < env._reward_height_thresh:
                # print('lifting')
                action = np.asarray([0., 0., 0.7])
                action = np.concatenate(
                    (action, np.asarray([0., 0., 0.])))
            else:
                # print('terminating')
                action = np.asarray([0., 0., 0.7])
                action = np.concatenate(
                    (action, np.asarray([0., 0., 0.7])))

            action += np.random.normal(scale=0.1, size=(6,))
            # print(action)
            obs, rew, done, info = env.step(action)
            if rew > 0:
                print('reward: {}'.format(rew))
            if done:
                break

