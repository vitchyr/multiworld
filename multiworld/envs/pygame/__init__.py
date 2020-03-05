from gym.envs.registration import register
import logging

LOGGER = logging.getLogger(__name__)
REGISTERED = False

import numpy as np
def axis_goal_sampler(env, batch_size):
    pos = np.random.uniform(env.obs_range.low, env.obs_range.high, (batch_size, 2))
    for idx in range(len(pos)):
        if np.random.random() > 0.5:
            pos[idx][0] = 0.0
        else:
            pos[idx][1] = 0.0
    return dict(
        state_desired_goal=pos,
        desired_goal=pos
    )

def half_axis_goal_sampler(env, batch_size):
    pos = np.random.uniform(env.obs_range.low / 2, env.obs_range.high / 2, (batch_size, 2))
    for idx in range(len(pos)):
        if np.random.random() > 0.5:
            pos[idx][0] = 0.0
        else:
            pos[idx][1] = 0.0
    return dict(
        state_desired_goal=pos,
        desired_goal=pos
    )


def full_goal_sampler(env, batch_size):
    pos = np.random.uniform(env.obs_range.low, env.obs_range.high, (batch_size, 2))
    return dict(
        state_desired_goal=pos,
        desired_goal=pos
    )

def bottom_corner_sampler(env, batch_size):
    low = np.array([0.5 * env.inner_wall_max_dist, 0.5 * env.inner_wall_max_dist])
    high = np.array([env.boundary_dist, env.boundary_dist])
    pos = [env._sample_position(low, high) for _ in range(batch_size)]
    pos = np.r_[pos].reshape(batch_size, 2)
    return dict(
        state_desired_goal=pos,
        desired_goal=pos
    )

def register_pygame_envs():
    global REGISTERED
    if REGISTERED:
        return
    REGISTERED = True
    LOGGER.info("Registering multiworld pygame gym environments")
    register(
        id='Point2DEnv-Train-Half-Axis-Eval-Everything-Images-v0',
        entry_point=point2d_image_train_half_axis_eval_all_v1,
        tags={
            'git-commit-hash': '78c9f9e',
            'author': 'vitchyr'
        },
    )
    register(
        id='Point2DEnv-Train-Axis-Eval-Everything-Images-v0',
        entry_point=point2d_image_train_axis_eval_all_v1,
        tags={
            'git-commit-hash': '78c9f9e',
            'author': 'vitchyr'
        },
    )
    register(
        id='Point2DEnv-Train-Half-Axis-Eval-Everything-Images-16-v0',
        entry_point=point2d_image_train_half_axis_eval_all_16_v1,
        tags={
            'git-commit-hash': '78c9f9e',
            'author': 'vitchyr'
        },
    )


    register(
        id='Point2DEnv-Train-Half-Axis-Eval-Everything-v0',
        entry_point='multiworld.envs.pygame.point2d:Point2DEnv',
        tags={
            'git-commit-hash': '166f0f3',
            'author': 'steven'
        },
        kwargs={
            'images_are_rgb': True,
            'target_radius': 1,
            'ball_radius': 1,
            'action_scale': 0.05,
            'render_onscreen': False,
            'fixed_reset': np.array([0, 0]),
            'eval_goal_sampler': full_goal_sampler,
            'expl_goal_sampler': half_axis_goal_sampler,
            'randomize_position_on_reset': False,
            'reward_type': 'dense_l1',
        },
    )

    register(
        id='Point2DEnv-Train-Axis-Eval-Everything-v0',
        entry_point='multiworld.envs.pygame.point2d:Point2DEnv',
        tags={
            'git-commit-hash': '166f0f3',
            'author': 'steven'
        },
        kwargs={
            'images_are_rgb': True,
            'target_radius': 1,
            'ball_radius': 1,
            'action_scale': 0.05,
            'render_onscreen': False,
            'fixed_reset': np.array([0, 0]),
            'eval_goal_sampler': full_goal_sampler,
            'expl_goal_sampler': axis_goal_sampler,
            'randomize_position_on_reset': False,
            'reward_type': 'dense_l1',
        },
    )

    register(
        id='Point2DEnv-Train-Everything-Eval-Everything-v0',
        entry_point='multiworld.envs.pygame.point2d:Point2DEnv',
        tags={
            'git-commit-hash': '9ebe203',
            'author': 'Vitchyr'
        },
        kwargs={
            'images_are_rgb': True,
            'target_radius': 1,
            'ball_radius': 1,
            'action_scale': 0.05,
            'render_onscreen': False,
            'fixed_reset': np.array([0, 0]),
            'eval_goal_sampler': full_goal_sampler,
            'expl_goal_sampler': full_goal_sampler,
            'randomize_position_on_reset': False,
            'reward_type': 'dense_l1',
        },
    )
    register(
        id='Point2DEnv-Wall-v0',
        entry_point='multiworld.envs.pygame.point2d:Point2DWallEnv',
        tags={
            'git-commit-hash': '166f0f3',
            'author': 'steven'
        },
        kwargs={
            'images_are_rgb': True,
            'target_radius': 0.5,
            'ball_radius': 0.5,
            'action_scale': 0.15,
            'render_onscreen': False,
            'fixed_reset': np.array([0, 0]),
            'use_fixed_reset_for_eval': True,
            'wall_shape': '-|',
            'wall_thickness': 0.5,
            'eval_goal_sampler': bottom_corner_sampler,
            'randomize_position_on_reset': True,
            'reward_type': 'dense_l1',
        },
    )
    register(
        id='Point2DLargeEnv-offscreen-v0',
        entry_point='multiworld.envs.pygame.point2d:Point2DEnv',
        tags={
            'git-commit-hash': '166f0f3',
            'author': 'Vitchyr'
        },
        kwargs={
            'images_are_rgb': True,
            'target_radius': 1,
            'ball_radius': 1,
            'render_onscreen': False,
        },
    )
    register(
        id='Point2DLargeEnv-onscreen-v0',
        entry_point='multiworld.envs.pygame.point2d:Point2DEnv',
        tags={
            'git-commit-hash': '166f0f3',
            'author': 'Vitchyr'
        },
        kwargs={
            'images_are_rgb': True,
            'target_radius': 1,
            'ball_radius': 1,
            'render_onscreen': True,
        },
    )
    register(
        id='Point2D-Box-Wall-v1',
        entry_point='multiworld.envs.pygame.point2d:Point2DWallEnv',
        tags={
            'git-commit-hash': '73c8823',
            'author': 'vitchyr'
        },
        kwargs={
            'action_scale': 1.,
            'wall_shape': 'box',
            'wall_thickness': 2.0,
            'render_size': 84,
            'render_onscreen': True,
            'images_are_rgb': True,
            'render_target': True,
        },
    )
    register(
        id='Point2D-Big-UWall-v1',
        entry_point='multiworld.envs.pygame.point2d:Point2DWallEnv',
        tags={
            'git-commit-hash': '73c8823',
            'author': 'vitchyr'
        },
        kwargs={
            'action_scale': 0.25,
            'wall_shape': 'big-u',
            'wall_thickness': 0.50,
            'render_size': 84,
            'images_are_rgb': True,
            'render_onscreen': True,
            'render_target': True,
        },
    )
    register(
        id='Point2D-Easy-UWall-v1',
        entry_point='multiworld.envs.pygame.point2d:Point2DWallEnv',
        tags={
            'git-commit-hash': '73c8823',
            'author': 'vitchyr'
        },
        kwargs={
            'action_scale': 0.25,
            'wall_shape': 'easy-u',
            'wall_thickness': 0.50,
            'render_size': 84,
            'images_are_rgb': True,
            'render_onscreen': True,
            'render_target': True,
        },
    )
    register(
        id='Point2DEnv-ImageFixedGoal-v0',
        entry_point=point2d_image_fixed_goal_v0,
        tags={
            'git-commit-hash': '2e92a51',
            'author': 'vitchyr'
        },
    )
    register(
        id='Point2DEnv-Image-v0',
        entry_point=point2d_image_v0,
        tags={
            'git-commit-hash': '78c9f9e',
            'author': 'vitchyr'
        },
    )


def point2d_image_fixed_goal_v0(**kwargs):
    from multiworld.core.image_env import ImageEnv
    from multiworld.envs.pygame.point2d import Point2DEnv
    from multiworld.core.flat_goal_env import FlatGoalEnv
    env = Point2DEnv(
        fixed_goal=(0, 0),
        images_are_rgb=True,
        render_onscreen=False,
        show_goal=True,
        ball_radius=2,
        render_size=8,
    )
    env = ImageEnv(env, imsize=env.render_size, transpose=True)
    env = FlatGoalEnv(env)
    return env


def point2d_image_v0(**kwargs):
    from multiworld.core.image_env import ImageEnv
    from multiworld.envs.pygame.point2d import Point2DEnv
    env = Point2DEnv(
        images_are_rgb=True,
        render_onscreen=False,
        show_goal=False,
        ball_radius=2,
        render_size=8,
    )
    env = ImageEnv(env, imsize=env.render_size, transpose=True)
    return env

def point2d_image_train_half_axis_eval_all_16_v1(**kwargs):
    from multiworld.core.image_env import ImageEnv
    from multiworld.envs.pygame.point2d import Point2DEnv
    env = Point2DEnv(
        images_are_rgb=True,
        render_onscreen=False,
        show_goal=False,
        render_size=16,
        target_radius=2,
        ball_radius=2,
        action_scale=0.05,
        fixed_reset=np.array([0, 0]),
        eval_goal_sampler=full_goal_sampler,
        expl_goal_sampler=half_axis_goal_sampler,
        randomize_position_on_reset=False,
        reward_type='dense_l1',
    )
    env = ImageEnv(env, imsize=env.render_size, normalize=True, transpose=True,
                   presample_goals_on_fly=True,
                   num_presampled_goals_on_fly=10000)
    return env

def point2d_image_train_half_axis_eval_all_v1(**kwargs):
    from multiworld.core.image_env import ImageEnv
    from multiworld.envs.pygame.point2d import Point2DEnv
    env = Point2DEnv(
        images_are_rgb=True,
        render_onscreen=False,
        show_goal=False,
        render_size=48,
        target_radius=2,
        ball_radius=2,
        action_scale=0.05,
        fixed_reset=np.array([0, 0]),
        eval_goal_sampler=full_goal_sampler,
        expl_goal_sampler=half_axis_goal_sampler,
        randomize_position_on_reset=False,
        reward_type='dense_l1',
    )
    env = ImageEnv(env, imsize=env.render_size, normalize=True, transpose=True,
                   presample_goals_on_fly=True,
                   num_presampled_goals_on_fly=10000)
    return env

def point2d_image_train_axis_eval_all_v1(**kwargs):
    from multiworld.core.image_env import ImageEnv
    from multiworld.envs.pygame.point2d import Point2DEnv
    env = Point2DEnv(
        images_are_rgb=True,
        render_onscreen=False,
        show_goal=False,
        render_size=48,
        target_radius=2,
        ball_radius=2,
        action_scale=0.05,
        fixed_reset=np.array([0, 0]),
        eval_goal_sampler=full_goal_sampler,
        expl_goal_sampler=axis_goal_sampler,
        randomize_position_on_reset=False,
        reward_type='dense_l1',
    )
    env = ImageEnv(env, imsize=env.render_size, normalize=True, transpose=True,
                   presample_goals_on_fly=True,
                   num_presampled_goals_on_fly=10000)
    return env
