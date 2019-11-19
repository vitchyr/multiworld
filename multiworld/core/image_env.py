import random

import cv2
import numpy as np
import warnings
from gym.spaces import Box, Dict

from multiworld.core.multitask_env import MultitaskEnv
from multiworld.core.wrapper_env import ProxyEnv
from multiworld.envs.env_util import concatenate_box_spaces
from multiworld.envs.env_util import get_stat_in_paths, create_stats_ordered_dict

class ImageEnv(ProxyEnv, MultitaskEnv):
    def __init__(
            self,
            wrapped_env,
            imsize=84,
            init_camera=None,
            transpose=False,
            grayscale=False,
            normalize=False,
            reward_type='wrapped_env',
            threshold=10,
            image_length=None,
            presampled_goals=None,
            non_presampled_goal_img_is_garbage=False,
            recompute_reward=True,
    ):
        """
        :param wrapped_env:
        :param imsize:
        :param init_camera:
        :param transpose:
        :param grayscale:
        :param normalize:
        :param reward_type:
        :param threshold:
        :param image_length:
        :param presampled_goals:
        :param non_presampled_goal_img_is_garbage: Set this option to True if
        you want to allow the code to work without presampled goals,
        but where the underlying env doesn't support set_to_goal. As the name,
        implies this will make it so that the goal image is garbage if you
        don't provide pre-sampled goals. The main use case is if you want to
        use an ImageEnv to pre-sample a bunch of goals.
        """
        self.quick_init(locals())
        super().__init__(wrapped_env)
        self.wrapped_env.hide_goal_markers = True
        self.imsize = imsize
        self.init_camera = init_camera
        self.transpose = transpose
        self.grayscale = grayscale
        self.normalize = normalize
        self.recompute_reward = recompute_reward
        self.non_presampled_goal_img_is_garbage = non_presampled_goal_img_is_garbage

        if image_length is not None:
            self.image_length = image_length
        else:
            if grayscale:
                self.image_length = self.imsize * self.imsize
            else:
                self.image_length = 3 * self.imsize * self.imsize

        self.channels = 1 if grayscale else 3

        # This is torch format rather than PIL image
        self.image_shape = (self.imsize, self.imsize)
        # Flattened past image queue
        # init camera
        if init_camera is not None:
            sim = self._wrapped_env.initialize_camera(init_camera)
            # viewer = mujoco_py.MjRenderContextOffscreen(sim, device_id=-1)
            # init_camera(viewer.cam)
            # sim.add_render_context(viewer)
        self._render_local = False
        img_space = Box(0, 1, (self.image_length,), dtype=np.float32)

        self._img_goal = img_space.sample() #has to be done for presampling

        spaces = self.wrapped_env.observation_space.spaces.copy()
        spaces['observation'] = img_space
        spaces['desired_goal'] = img_space
        spaces['achieved_goal'] = img_space
        spaces['image_observation'] = img_space
        spaces['image_desired_goal'] = img_space
        spaces['image_achieved_goal'] = img_space

        self.return_image_proprio = False
        if 'proprio_observation' in spaces.keys():
            self.return_image_proprio = True
            spaces['image_proprio_observation'] = concatenate_box_spaces(
                spaces['image_observation'],
                spaces['proprio_observation']
            )
            spaces['image_proprio_desired_goal'] = concatenate_box_spaces(
                spaces['image_desired_goal'],
                spaces['proprio_desired_goal']
            )
            spaces['image_proprio_achieved_goal'] = concatenate_box_spaces(
                spaces['image_achieved_goal'],
                spaces['proprio_achieved_goal']
            )
        self.observation_space = Dict(spaces)
        self.action_space = self.wrapped_env.action_space
        self.reward_type = reward_type
        self.threshold = threshold
        self._presampled_goals = presampled_goals
        self.dummy = False
        if self._presampled_goals is None:
            self.num_goals_presampled = 0
        else:
            self.num_goals_presampled = presampled_goals[random.choice(list(presampled_goals))].shape[0]

    def step(self, action):
        obs, reward, done, info = self.wrapped_env.step(action)
        new_obs = self._update_obs(obs)
        if self.recompute_reward:
            reward = self.compute_reward(action, new_obs)
        self._update_info(info, new_obs)
        return new_obs, reward, done, info

    def _update_info(self, info, obs):
        info['image_dist'] = 0
        info['image_success'] = 0

    def set_presampled_goals(self, goals):
        self._presampled_goals = goals
        if goals is not None:
            self.num_goals_presampled = len(goals['image_desired_goal'])

    def reset(self):
        obs = self.wrapped_env.reset()
        if self.num_goals_presampled > 0:
            goal = self.sample_goal()
            self._img_goal = goal['image_desired_goal']
            self.wrapped_env.set_goal(goal)
            for key in goal:
                obs[key] = goal[key]
        elif self.non_presampled_goal_img_is_garbage:
            # This is use mainly for debugging or pre-sampling goals.
            self._img_goal = self._get_flat_img()
        else:
            env_state = self.wrapped_env.get_env_state()
            self.wrapped_env.set_to_goal(self.wrapped_env.get_goal())
            self._img_goal = self._get_flat_img()
            self.wrapped_env.set_env_state(env_state)

        return self._update_obs(obs)

    def _get_obs(self):
        return self._update_obs(self.wrapped_env._get_obs())

    def _update_obs(self, obs):
        img_obs = self._get_flat_img()
        obs['image_observation'] = img_obs
        obs['image_desired_goal'] = self._img_goal
        obs['image_achieved_goal'] = img_obs
        obs['observation'] = img_obs
        obs['desired_goal'] = self._img_goal
        obs['achieved_goal'] = img_obs

        if self.return_image_proprio:
            obs['image_proprio_observation'] = np.concatenate(
                (obs['image_observation'], obs['proprio_observation'])
            )
            obs['image_proprio_desired_goal'] = np.concatenate(
                (obs['image_desired_goal'], obs['proprio_desired_goal'])
            )
            obs['image_proprio_achieved_goal'] = np.concatenate(
                (obs['image_achieved_goal'], obs['proprio_achieved_goal'])
            )

        return obs

    def merge_frames(self, frame1, frame2):
        frame = frame1.copy()
        for key in [
            'image_observation',
            'image_desired_goal',
            'image_achieved_goal',
            'observation',
            'desired_goal',
            'achieved_goal',
            'image_proprio_observation',
            'image_proprio_desired_goal',
            'image_proprio_achieved_goal',
        ]:
            if key in frame1 and key in frame2:
                frame[key] = np.concatenate((frame1[key], frame2[key]))
        return frame

    def _get_flat_img(self):
        if self.dummy:
            # return np.zeros(1)
            image_obs = np.zeros(self.image_length)
            return image_obs
        # returns the image as a torch format np array
        image_obs = self._wrapped_env.get_image(
            width=self.imsize,
            height=self.imsize,
        )
        return self.transform_image(image_obs)

    def state_to_image(self, state):
        images = self.states_to_images(state[None])
        if images is not None:
            return images[0]
        else:
            return None

    def states_to_images(self, states):
        from multiworld.envs.mujoco.locomotion.wheeled_car import WheeledCarEnv
        from multiworld.envs.mujoco.classic_mujoco.ant_maze import AntMazeEnv
        from multiworld.envs.mujoco.sawyer_xyz.sawyer_push_nips import SawyerPushAndReachXYEnv
        from multiworld.envs.mujoco.sawyer_xyz.sawyer_pick_and_place import (
            SawyerPickAndPlaceEnvYZ,
            SawyerPickAndPlaceEnv,
        )

        if isinstance(self._wrapped_env, WheeledCarEnv):
            state_dim = len(self.observation_space.spaces['state_observation'].low)
            orig_batch_size = states.shape[0]
            states = states.reshape(-1, state_dim)

            states = self._wrapped_env.valid_states(states)
            pre_state = self.wrapped_env.get_env_state()

            batch_size = states.shape[0]
            image_dim = self.image_length
            imgs = np.zeros((batch_size, image_dim))
            for i in range(batch_size):
                self.wrapped_env.set_to_goal({"state_desired_goal": states[i]})
                imgs[i, :] = self._get_flat_img()
                # if batch_size > 1:
                #     img = imgs[i, :]
                #     img = img.reshape(3, 84, 84).transpose((1, 2, 0))
                #     img = img[::, :, ::-1]
                #     cv2.imshow('img', img)
                #     cv2.waitKey(1)
            self.wrapped_env.set_env_state(pre_state)
            imgs = imgs.reshape(orig_batch_size, -1)
            return imgs
        elif isinstance(self._wrapped_env, AntMazeEnv):
            state_dim = len(self.observation_space.spaces['state_observation'].low)
            orig_batch_size = states.shape[0]
            states = states.reshape(-1, state_dim)

            pre_state = self.wrapped_env.get_env_state()

            batch_size = states.shape[0]
            image_dim = self.image_length
            imgs = np.zeros((batch_size, image_dim))
            for i in range(batch_size):
                self.wrapped_env.set_to_goal({"state_desired_goal": states[i]})
                imgs[i, :] = self._get_flat_img()
            self.wrapped_env.set_env_state(pre_state)
            imgs = imgs.reshape(orig_batch_size, -1)
            return imgs
        elif isinstance(self._wrapped_env, SawyerPickAndPlaceEnvYZ):
            return self.wrapped_env.states_to_images(states)

        elif isinstance(self._wrapped_env, SawyerPushAndReachXYEnv):
            batch_size = states.shape[0]
            imgs = np.zeros((batch_size, self.image_length))
            for i in range(batch_size):
                imgs[i, :] = self.transform_image(self._wrapped_env.get_image_plt(state=states[i]))
            return imgs

        return None

    def transform_image(self, img):
        if img is None:
            return None

        if self._render_local:
            cv2.imshow('env', img)
            cv2.waitKey(1)
        if self.grayscale:
            from PIL import Image
            img = Image.fromarray(img).convert('L')
            img = np.array(img)
        if self.normalize:
            img = img / 255.0
        if self.transpose:
            if img.ndim == 3:
                img = img.transpose((2, 0, 1))
            else:
                img = img.transpose()
        return img.flatten()

    def render(self):
        self.wrapped_env.render()

    def enable_render(self):
        self._render_local = True

    """
    Multitask functions
    """
    def get_goal(self):
        goal = self.wrapped_env.get_goal()
        goal['desired_goal'] = self._img_goal
        goal['image_desired_goal'] = self._img_goal
        return goal

    def set_goal(self, goal):
        ''' Assume goal contains both image_desired_goal and any goals required for wrapped envs'''
        self._img_goal = goal['image_desired_goal']
        self.wrapped_env.set_goal(goal)

    def sample_goals(self, batch_size):
        if self.num_goals_presampled > 0:
            idx = np.random.randint(0, self.num_goals_presampled, batch_size)
            sampled_goals = {
                k: v[idx] for k, v in self._presampled_goals.items()
            }
            return sampled_goals
        if batch_size > 1:
            warnings.warn("Sampling goal images is slow")
        img_goals = np.zeros((batch_size, self.image_length))
        goals = self.wrapped_env.sample_goals(batch_size)
        pre_state = self.wrapped_env.get_env_state()
        for i in range(batch_size):
            goal = self.unbatchify_dict(goals, i)
            self.wrapped_env.set_to_goal(goal)
            img = self._get_flat_img()
            img_goals[i, :] = img
        self.wrapped_env.set_env_state(pre_state)
        goals['desired_goal'] = img_goals
        goals['image_desired_goal'] = img_goals

        return goals

    def compute_rewards(self, actions, obs, prev_obs=None, reward_type=None):
        if reward_type is None:
            reward_type = self.reward_type

        if reward_type=='wrapped_env':
            return self.wrapped_env.compute_rewards(actions, obs, prev_obs=prev_obs)
        elif reward_type=='image_distance':
            achieved_goals = obs['image_achieved_goal']
            desired_goals = obs['image_desired_goal']
            dist = np.linalg.norm(achieved_goals - desired_goals, axis=1)
            return -dist
        elif reward_type=='image_sparse':
            achieved_goals = obs['image_achieved_goal']
            desired_goals = obs['image_desired_goal']
            dist = np.linalg.norm(achieved_goals - desired_goals, axis=1)
            return -(dist > self.threshold).astype(float)
        elif reward_type == 'image_distance_vectorized':
            achieved_goals = obs['image_achieved_goal']
            desired_goals = obs['image_desired_goal']
            dist = np.abs(achieved_goals - desired_goals)
            return -dist
        else:
            return self.wrapped_env.compute_rewards(actions, obs, prev_obs=prev_obs,
                                                    reward_type=reward_type)

    def get_diagnostics(self, paths, **kwargs):
        statistics = self.wrapped_env.get_diagnostics(paths, **kwargs)
        for stat_name_in_paths in ["image_dist", "image_success"]:
            stats = get_stat_in_paths(paths, 'env_infos', stat_name_in_paths)
            statistics.update(create_stats_ordered_dict(
                stat_name_in_paths,
                stats,
                always_show_all_stats=True,
            ))
            final_stats = [s[-1] for s in stats]
            statistics.update(create_stats_ordered_dict(
                "Final " + stat_name_in_paths,
                final_stats,
                always_show_all_stats=True,
            ))
        return statistics

    def update_subgoals(self, latent_subgoals, latent_subgoals_noisy):
        import warnings
        warnings.warn("Not implemented for image env")

def normalize_image(image):
    assert image.dtype == np.uint8
    return np.float32(image) / 255.0

def unormalize_image(image):
    assert image.dtype != np.uint8
    return np.uint8(image * 255.0)

def get_image_presampled_goals(image_env, num_goals_presampled):
    import time
    print("sampling image goals from image_env:", num_goals_presampled)
    t = time.time()
    image_goals = image_env.sample_goals(num_goals_presampled)
    print("total time:", time.time() - t)
    return image_goals
