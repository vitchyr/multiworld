from gym.envs.registration import register
import logging

LOGGER = logging.getLogger(__name__)
REGISTERED = False


def register_pygame_envs():
    global REGISTERED
    if REGISTERED:
        return
    REGISTERED = True
    LOGGER.info("Registering multiworld pygame gym environments")
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
