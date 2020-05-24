from gym.envs.registration import register
import logging

from multiworld.envs.pygame.pick_and_place import (
    PickAndPlaceEnv,
    PickAndPlace1DEnv,
)

LOGGER = logging.getLogger(__name__)
REGISTERED = False


def register_pygame_envs():
    global REGISTERED
    if REGISTERED:
        return
    REGISTERED = True
    LOGGER.info("Registering multiworld pygame gym environments")
    register_pnp_envs()
    register(
        id='Point2DLargeEnv-v1',
        entry_point='multiworld.envs.pygame.point2d:Point2DEnv',
        tags={
            'git-commit-hash': '4efe2be',
            'author': 'Vitchyr'
        },
        kwargs={
            'images_are_rgb': True,
            'target_radius': 1,
            'ball_radius': 1,
            'render_onscreen': False,
            'show_goal': False,
            'get_image_base_render_size': (48, 48),
        },
    )
    register(
        id='Point2DEasyEnv-v1',
        entry_point='multiworld.envs.pygame.point2d:Point2DEnv',
        tags={
            'author': 'Ashvin'
        },
        kwargs={
            'images_are_rgb': True,
            'target_radius': 1,
            'ball_radius': 2,
            'render_onscreen': False,
            'show_goal': False,
            'get_image_base_render_size': (48, 48),
            'bg_color': 'white',
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
            'wall_color': 'black',
            'bg_color': 'white',
        },
    )
    register(
        id='Point2D-Easy-UWall-v2',
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
            'wall_color': 'white',
            'bg_color': 'black',
            'images_are_rgb': True,
            'render_onscreen': False,
            'show_goal': False,
            'get_image_base_render_size': (48, 48),
        },
    )
    register(
        id='Point2D-Easy-UWall-Hard-Init-v2',
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
            'wall_color': 'white',
            'bg_color': 'black',
            'images_are_rgb': True,
            'render_onscreen': False,
            'show_goal': False,
            'fixed_init_position': (0, -2),
            'randomize_position_on_reset': False,
            'fixed_goal': (0, 3),
            'get_image_base_render_size': (48, 48),
        },
    )
    register(
        id='Point2D-FlatWall-v2',
        entry_point='multiworld.envs.pygame.point2d:Point2DWallEnv',
        kwargs={
            'action_scale': 0.25,
            'wall_shape': '---',
            'wall_thickness': 0.50,
            'render_size': 84,
            'wall_color': 'white',
            'bg_color': 'black',
            'images_are_rgb': True,
            'render_onscreen': False,
            'show_goal': False,
            'get_image_base_render_size': (48, 48),
        },
    )
    register(
        id='Point2D-FlatWall-Hard-Init-v2',
        entry_point='multiworld.envs.pygame.point2d:Point2DWallEnv',
        kwargs={
            'action_scale': 0.25,
            'wall_shape': '---',
            'wall_thickness': 0.50,
            'render_size': 84,
            'wall_color': 'white',
            'bg_color': 'black',
            'images_are_rgb': True,
            'render_onscreen': False,
            'show_goal': False,
            'fixed_init_position': (0, -2),
            'randomize_position_on_reset': False,
            'fixed_goal': (0, 3),
            'get_image_base_render_size': (48, 48),
        },
    )
    register(
        id='Point2D-ImageFixedGoal-v0',
        entry_point=point2d_image_fixed_goal_v0,
        tags={
            'git-commit-hash': '2e92a51',
            'author': 'vitchyr'
        },
    )
    register(
        id='Point2D-Image-v0',
        entry_point=point2d_image_v0,
        tags={
            'git-commit-hash': '78c9f9e',
            'author': 'vitchyr'
        },
    )


def register_pnp_envs():
    shared_settings = dict(
        # Environment dynamics
        action_scale=1.0,
        ball_radius=1.,
        boundary_dist=4,
        object_radius=0.50,
        min_grab_distance=0.5,
        walls=None,
        # Rewards
        action_l2norm_penalty=0,
        reward_type="dense_l1",
        success_threshold=0.60,
        # Reset settings
        fixed_goal=None,
        # Visualization settings
        images_are_rgb=True,
        render_dt_msec=0,
        render_onscreen=False,
        render_size=84,
        show_goal=False,
        get_image_base_render_size=(48, 48),
        # Goal sampling
        goal_samplers=None,
        goal_sampling_mode='random',
        num_presampled_goals=10000,
        object_reward_only=True,
    )
    for env_id, extra_settings in [
        (
                'FiveObject-PickAndPlace-RandomInit-1D-v1',
                dict(num_objects=5, init_position_strategy='random'),
        ),
        (
                'FourObject-PickAndPlace-RandomInit-1D-v1',
                dict(num_objects=4, init_position_strategy='random'),
        ),
        (
                'ThreeObject-PickAndPlace-RandomInit-1D-v1',
                dict(num_objects=3, init_position_strategy='random'),
        ),
        (
                'TwoObject-PickAndPlace-RandomInit-1D-v1',
                dict(num_objects=2, init_position_strategy='random'),
        ),
        (
                'OneObject-PickAndPlace-RandomInit-1D-v1',
                dict(num_objects=1, init_position_strategy='random'),
        ),
        (
                'FiveObject-PickAndPlace-OriginInit-1D-v1',
                dict(num_objects=5, init_position_strategy='fixed'),
        ),
        (
                'FourObject-PickAndPlace-OriginInit-1D-v1',
                dict(num_objects=4, init_position_strategy='fixed'),
        ),
        (
                'ThreeObject-PickAndPlace-OriginInit-1D-v1',
                dict(num_objects=3, init_position_strategy='fixed'),
        ),
        (
                'TwoObject-PickAndPlace-OriginInit-1D-v1',
                dict(num_objects=2, init_position_strategy='fixed'),
        ),
        (
                'OneObject-PickAndPlace-OriginInit-1D-v1',
                dict(num_objects=1, init_position_strategy='fixed'),
        ),
        (
                'FiveObject-PickAndPlace-OnRandomObjectInit-1D-v1',
                dict(num_objects=5, init_position_strategy='on_random_object'),
        ),
        (
                'FourObject-PickAndPlace-OnRandomObjectInit-1D-v1',
                dict(num_objects=4, init_position_strategy='on_random_object'),
        ),
        (
                'ThreeObject-PickAndPlace-OnRandomObjectInit-1D-v1',
                dict(num_objects=3, init_position_strategy='on_random_object'),
        ),
        (
                'TwoObject-PickAndPlace-OnRandomObjectInit-1D-v1',
                dict(num_objects=2, init_position_strategy='on_random_object'),
        ),
        (
                'OneObject-PickAndPlace-OnRandomObjectInit-1D-v1',
                dict(num_objects=1, init_position_strategy='on_random_object'),
        ),
    ]:
        new_kwargs = shared_settings.copy()
        new_kwargs.update(extra_settings)
        register(
            id=env_id,
            entry_point=PickAndPlace1DEnv,
            kwargs=new_kwargs,
        )
    for env_id, extra_settings in [
        (
                'FiveObject-PickAndPlace-RandomInit-2D-v1',
                dict(num_objects=5, init_position_strategy='random'),
        ),
        (
                'FourObject-PickAndPlace-RandomInit-2D-v1',
                dict(num_objects=4, init_position_strategy='random'),
        ),
        (
                'ThreeObject-PickAndPlace-RandomInit-2D-v1',
                dict(num_objects=3, init_position_strategy='random'),
        ),
        (
                'TwoObject-PickAndPlace-RandomInit-2D-v1',
                dict(num_objects=2, init_position_strategy='random'),
        ),
        (
                'OneObject-PickAndPlace-RandomInit-2D-v1',
                dict(num_objects=1, init_position_strategy='random'),
        ),
        (
                'FiveObject-PickAndPlace-OriginInit-2D-v1',
                dict(num_objects=5, init_position_strategy='fixed'),
        ),
        (
                'FourObject-PickAndPlace-OriginInit-2D-v1',
                dict(num_objects=4, init_position_strategy='fixed'),
        ),
        (
                'ThreeObject-PickAndPlace-OriginInit-2D-v1',
                dict(num_objects=3, init_position_strategy='fixed'),
        ),
        (
                'TwoObject-PickAndPlace-OriginInit-2D-v1',
                dict(num_objects=2, init_position_strategy='fixed'),
        ),
        (
                'OneObject-PickAndPlace-OriginInit-2D-v1',
                dict(num_objects=1, init_position_strategy='fixed'),
        ),
        (
                'FiveObject-PickAndPlace-OnRandomObjectInit-2D-v1',
                dict(num_objects=5, init_position_strategy='on_random_object'),
        ),
        (
                'FourObject-PickAndPlace-OnRandomObjectInit-2D-v1',
                dict(num_objects=4, init_position_strategy='on_random_object'),
        ),
        (
                'ThreeObject-PickAndPlace-OnRandomObjectInit-2D-v1',
                dict(num_objects=3, init_position_strategy='on_random_object'),
        ),
        (
                'TwoObject-PickAndPlace-OnRandomObjectInit-2D-v1',
                dict(num_objects=2, init_position_strategy='on_random_object'),
        ),
        (
                'OneObject-PickAndPlace-OnRandomObjectInit-2D-v1',
                dict(num_objects=1, init_position_strategy='on_random_object'),
        ),
    ]:
        new_kwargs = shared_settings.copy()
        new_kwargs.update(extra_settings)
        register(
            id=env_id,
            entry_point=PickAndPlaceEnv,
            kwargs=new_kwargs,
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
