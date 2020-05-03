from gym.envs.registration import register
import logging
import numpy as np

from multiworld.envs.pygame.pick_and_place import (
    PickAndPlaceEnv,
    PickAndPlace1DEnv
)

LOGGER = logging.getLogger(__name__)
REGISTERED = False


def register_pygame_envs():
    global REGISTERED
    if REGISTERED:
        return
    REGISTERED = True
    LOGGER.info("Registering multiworld pygame gym environments")
    register(
        id='FiveObjectPickAndPlaceRandomInit1DEnv-v0',
        entry_point=PickAndPlace1DEnv,
        tags={
            'git-commit-hash': 'f2c7f9f',
            'author': 'vitchyr'
        },
        kwargs=dict(
            num_objects=5,
            # Environment dynamics
            action_scale=1.0,
            ball_radius=.75,
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
            randomize_position_on_reset=True,
            # Visualization settings
            images_are_rgb=True,
            render_dt_msec=0,
            render_onscreen=False,
            render_size=84,
            show_goal=False,
            # Goal sampling
            goal_samplers=None,
            goal_sampling_mode='random',
            num_presampled_goals=10000,
            object_reward_only=True,
        ),
    )


    register(
        id='TwoObjectPickAndPlaceRandomInit1DEnv-v0',
        entry_point=PickAndPlace1DEnv,
        tags={
            'git-commit-hash': 'f2c7f9f',
            'author': 'vitchyr'
        },
        kwargs=dict(
            num_objects=2,
            # Environment dynamics
            action_scale=1.0,
            ball_radius=.75,
            boundary_dist=4,
            object_radius=0.50,
            min_grab_distance=0.75,
            walls=None,
            # Rewards
            action_l2norm_penalty=0,
            reward_type="dense_l1",
            success_threshold=0.60,
            # Reset settings
            fixed_goal=None,
            randomize_position_on_reset=True,
            # fixed_init_position=np.array([0, 0, 0, 2, 0, -2, 0, 4, 0, -4]),
            # Visualization settings
            images_are_rgb=True,
            render_dt_msec=0,
            render_onscreen=False,
            render_size=84,
            show_goal=False,
            # Goal sampling
            goal_samplers=None,
            goal_sampling_mode='random',
            num_presampled_goals=10000,
        ),
    )


    register(
        id='ThreeObjectPickAndPlaceRandomInit1DEnv-v0',
        entry_point=PickAndPlace1DEnv,
        tags={
            'git-commit-hash': 'f2c7f9f',
            'author': 'vitchyr'
        },
        kwargs=dict(
            num_objects=3,
            # Environment dynamics
            action_scale=1.0,
            ball_radius=.75,
            boundary_dist=4,
            object_radius=0.50,
            min_grab_distance=0.75,
            walls=None,
            # Rewards
            action_l2norm_penalty=0,
            reward_type="dense_l1",
            success_threshold=0.60,
            # Reset settings
            fixed_goal=None,
            randomize_position_on_reset=True,
            # fixed_init_position=np.array([0, 0, 0, 2, 0, -2, 0, 4, 0, -4]),
            # Visualization settings
            images_are_rgb=True,
            render_dt_msec=0,
            render_onscreen=False,
            render_size=84,
            show_goal=False,
            # Goal sampling
            goal_samplers=None,
            goal_sampling_mode='random',
            num_presampled_goals=10000,
            object_reward_only=True,
        ),
    )


    register(
        id='FourObjectPickAndPlace1DEnv-v0',
        entry_point=PickAndPlace1DEnv,
        tags={
            'git-commit-hash': 'f2c7f9f',
            'author': 'vitchyr'
        },
        kwargs=dict(
            num_objects=4,
            # Environment dynamics
            action_scale=1.0,
            ball_radius=.75,
            boundary_dist=4,
            object_radius=0.50,
            min_grab_distance=0.75,
            walls=None,
            # Rewards
            action_l2norm_penalty=0,
            reward_type="dense_l1",
            success_threshold=0.60,
            # Reset settings
            fixed_goal=None,
            randomize_position_on_reset=False,
            fixed_init_position=np.array([0, 0, 0, 2, 0, -2, 0, 4, 0, -4]),
            # Visualization settings
            images_are_rgb=True,
            render_dt_msec=0,
            render_onscreen=False,
            render_size=84,
            show_goal=False,
            # Goal sampling
            goal_samplers=None,
            goal_sampling_mode='random',
            num_presampled_goals=10000,
        ),
    )


    register(
        id='FiveObjectPickAndPlace1DEnv-v0',
        entry_point=PickAndPlace1DEnv,
        tags={
            'git-commit-hash': 'f2c7f9f',
            'author': 'vitchyr'
        },
        kwargs=dict(
            num_objects=5,
            # Environment dynamics
            action_scale=1.0,
            ball_radius=.75,
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
            randomize_position_on_reset=False,
            fixed_init_position=np.array([0, 0, 0, 0, 0, 2, 0, -2, 0, 4, 0, -4]),
            # Visualization settings
            images_are_rgb=True,
            render_dt_msec=0,
            render_onscreen=False,
            render_size=84,
            show_goal=False,
            # Goal sampling
            goal_samplers=None,
            goal_sampling_mode='random',
            num_presampled_goals=10000,
        ),
    )

    register(
        id='TwoObjectPickAndPlace1DSpreadEnv-v0',
        entry_point=PickAndPlace1DEnv,
        tags={
            'git-commit-hash': 'f2c7f9f',
            'author': 'vitchyr'
        },
        kwargs=dict(
            num_objects=2,
            # Environment dynamics
            action_scale=1.0,
            ball_radius=.75,
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
            fixed_init_position=np.array([0, 0, 0, 2, 0, -2]),
            randomize_position_on_reset=False,
            # Visualization settings
            images_are_rgb=True,
            render_dt_msec=0,
            render_onscreen=False,
            render_size=84,
            show_goal=False,
            # Goal sampling
            goal_samplers=None,
            goal_sampling_mode='random',
            num_presampled_goals=10000,
        ),
    )



    register(
        id='TwoObjectPickAndPlace1DEnv-v0',
        entry_point=PickAndPlace1DEnv,
        tags={
            'git-commit-hash': 'f2c7f9f',
            'author': 'vitchyr'
        },
        kwargs=dict(
            num_objects=2,
            # Environment dynamics
            action_scale=1.0,
            ball_radius=.75,
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
            fixed_init_position=None,
            randomize_position_on_reset=False,
            # Visualization settings
            images_are_rgb=True,
            render_dt_msec=0,
            render_onscreen=False,
            render_size=84,
            show_goal=False,
            # Goal sampling
            goal_samplers=None,
            goal_sampling_mode='random',
            num_presampled_goals=10000,
        ),
    )
    register(
        id='FourObjectPickAndPlace2DRandomInitEnv-v0',
        entry_point=PickAndPlaceEnv,
        tags={
            'git-commit-hash': 'f2c7f9f',
            'author': 'vitchyr'
        },
        kwargs=dict(
            num_objects=4,
            # Environment dynamics
            action_scale=1.0,
            ball_radius=.75,
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
            fixed_init_position=None,
            randomize_position_on_reset=True,
            # Visualization settings
            images_are_rgb=True,
            render_dt_msec=0,
            render_onscreen=False,
            render_size=84,
            show_goal=False,
            # Goal sampling
            goal_samplers=None,
            goal_sampling_mode='random',
            num_presampled_goals=10000,
            object_reward_only=True,
        ),
    )

    register(
        id='TwoObjectPickAndPlace2DEnv-v0',
        entry_point=PickAndPlaceEnv,
        tags={
            'git-commit-hash': 'f2c7f9f',
            'author': 'vitchyr'
        },
        kwargs=dict(
            num_objects=2,
            # Environment dynamics
            action_scale=1.0,
            ball_radius=.75,
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
            fixed_init_position=None,
            randomize_position_on_reset=False,
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
        ),
    )
    register(
        id='OneObjectPickAndPlace2DEnv-v0',
        entry_point=PickAndPlaceEnv,
        tags={
            'git-commit-hash': '518675c',
            'author': 'vitchyr'
        },
        kwargs=dict(
            num_objects=1,
            # Environment dynamics
            action_scale=1.0,
            ball_radius=.75,
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
            fixed_init_position=None,
            randomize_position_on_reset=False,
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
        ),
    )
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
