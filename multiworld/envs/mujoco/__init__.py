import gym
from gym.envs.registration import register
import logging

LOGGER = logging.getLogger(__name__)
REGISTERED = False


def register_mujoco_envs():
    global REGISTERED
    if REGISTERED:
        return
    REGISTERED = True
    LOGGER.info("Registering multiworld mujoco gym environments")
    from multiworld.envs.mujoco.cameras import (
        sawyer_init_camera_zoomed_in
    )
    register_classic_mujoco_envs()
    """
    Reaching tasks
    """

    register(
        id='SawyerReachXYEnv-v1',
        entry_point='multiworld.envs.mujoco.sawyer_xyz.sawyer_reach:SawyerReachXYEnv',
        tags={
            'git-commit-hash': '2d95c75',
            'author': 'murtaza'
        },
        kwargs={
            'hide_goal_markers': True,
            'norm_order': 2,
        },
    )

    register(
        id='SawyerReachXYZEnv-v0',
        entry_point='multiworld.envs.mujoco.sawyer_xyz.sawyer_reach:SawyerReachXYZEnv',
        tags={
            'git-commit-hash': '7b3113b',
            'author': 'vitchyr'
        },
        kwargs={
            'hide_goal_markers': False,
            'norm_order': 2,
        },
    )

    register(
        id='SawyerReachXYZEnv-v1',
        entry_point='multiworld.envs.mujoco.sawyer_xyz.sawyer_reach:SawyerReachXYZEnv',
        tags={
            'git-commit-hash': 'bea5de',
            'author': 'murtaza'
        },
        kwargs={
            'hide_goal_markers': True,
            'norm_order': 2,
        },
    )

    register(
        id='SawyerReachTorqueEnv-v0',
        entry_point='multiworld.envs.mujoco.sawyer_torque.sawyer_torque_reach:SawyerReachTorqueEnv',
        tags={
            'git-commit-hash': '0892abd',
            'author': 'murtaza'
        },
        kwargs={
            'keep_vel_in_obs': True,
            'use_safety_box': False,
            'torque_action_scale':100,
            'gripper_action_scale':1,
        },
    )

    register(
        id='Image48SawyerReachXYEnv-v1',
        entry_point=create_image_48_sawyer_reach_xy_env_v1,
        tags={
            'git-commit-hash': '2d95c75',
            'author': 'murtaza'
        },
    )
    register(
        id='Image84SawyerReachXYEnv-v1',
        entry_point=create_image_84_sawyer_reach_xy_env_v1,
        tags={
            'git-commit-hash': '2d95c75',
            'author': 'murtaza'
        },
    )


    """
    Pushing Tasks, XY
    """

    register(
        id='SawyerPushAndReachEnvEasy-v0',
        entry_point='multiworld.envs.mujoco.sawyer_xyz'
                    '.sawyer_push_and_reach_env:SawyerPushAndReachXYEnv',
        tags={
            'git-commit-hash': 'ddd73dc',
            'author': 'murtaza',
        },
        kwargs=dict(
            goal_low=(-0.15, 0.4, 0.02, -.1, .45),
            goal_high=(0.15, 0.7, 0.02, .1, .65),
            puck_low=(-.1, .45),
            puck_high=(.1, .65),
            hand_low=(-0.15, 0.4, 0.02),
            hand_high=(0.15, .7, 0.02),
            norm_order=2,
            xml_path='sawyer_xyz/sawyer_push_puck.xml',
            reward_type='state_distance',
            reset_free=False,
            clamp_puck_on_step=True,
        )
    )

    register(
        id='SawyerPushAndReachEnvMedium-v0',
        entry_point='multiworld.envs.mujoco.sawyer_xyz'
                    '.sawyer_push_and_reach_env:SawyerPushAndReachXYEnv',
        tags={
            'git-commit-hash': 'ddd73dc',
            'author': 'murtaza',
        },
        kwargs=dict(
            goal_low=(-0.2, 0.35, 0.02, -.15, .4),
            goal_high=(0.2, 0.75, 0.02, .15, .7),
            puck_low=(-.15, .4),
            puck_high=(.15, .7),
            hand_low=(-0.2, 0.35, 0.05),
            hand_high=(0.2, .75, 0.3),
            norm_order=2,
            xml_path='sawyer_xyz/sawyer_push_puck.xml',
            reward_type='state_distance',
            reset_free=False,
            clamp_puck_on_step=True,
        )
    )

    register(
        id='SawyerPushAndReachEnvHard-v0',
        entry_point='multiworld.envs.mujoco.sawyer_xyz'
                    '.sawyer_push_and_reach_env:SawyerPushAndReachXYEnv',
        tags={
            'git-commit-hash': 'ddd73dc',
            'author': 'murtaza',
        },
        kwargs=dict(
            goal_low=(-0.25, 0.3, 0.02, -.2, .35),
            goal_high=(0.25, 0.8, 0.02, .2, .75),
            puck_low=(-.2, .35),
            puck_high=(.2, .75),
            hand_low=(-0.25, 0.3, 0.02),
            hand_high=(0.25, .8, 0.02),
            norm_order=2,
            xml_path='sawyer_xyz/sawyer_push_puck.xml',
            reward_type='state_distance',
            reset_free=False,
            clamp_puck_on_step=True,
        )
    )

    """
    Pushing tasks, XY, Arena
    """
    register(
        id='SawyerPushAndReachArenaEnv-v0',
        entry_point='multiworld.envs.mujoco.sawyer_xyz'
                    '.sawyer_push_and_reach_env:SawyerPushAndReachXYEnv',
        tags={
            'git-commit-hash': 'dea1627',
            'author': 'murtaza',
        },
        kwargs=dict(
            goal_low=(-0.25, 0.3, 0.02, -.2, .4),
            goal_high=(0.25, 0.875, 0.02, .2, .8),
            puck_low=(-.4, .2),
            puck_high=(.4, 1),
            hand_low=(-0.28, 0.3, 0.05),
            hand_high=(0.28, 0.9, 0.3),
            norm_order=2,
            xml_path='sawyer_xyz/sawyer_push_puck_arena.xml',
            reward_type='state_distance',
            reset_free=False,
        )
    )

    register(
        id='SawyerPushAndReachArenaResetFreeEnv-v0',
        entry_point='multiworld.envs.mujoco.sawyer_xyz'
                    '.sawyer_push_and_reach_env:SawyerPushAndReachXYEnv',
        tags={
            'git-commit-hash': 'dea1627',
            'author': 'murtaza',
        },
        kwargs=dict(
            goal_low=(-0.25, 0.3, 0.02, -.2, .4),
            goal_high=(0.25, 0.875, 0.02, .2, .8),
            puck_low=(-.4, .2),
            puck_high=(.4, 1),
            hand_low=(-0.28, 0.3, 0.05),
            hand_high=(0.28, 0.9, 0.3),
            norm_order=2,
            xml_path='sawyer_xyz/sawyer_push_puck_arena.xml',
            reward_type='state_distance',
            reset_free=True,
        )
    )

    register(
        id='SawyerPushAndReachSmallArenaEnv-v0',
        entry_point='multiworld.envs.mujoco.sawyer_xyz'
                    '.sawyer_push_and_reach_env:SawyerPushAndReachXYEnv',
        tags={
            'git-commit-hash': '7256aaf',
            'author': 'murtaza',
        },
        kwargs=dict(
            goal_low=(-0.15, 0.4, 0.02, -.1, .5),
            goal_high=(0.15, 0.75, 0.02, .1, .7),
            puck_low=(-.3, .25),
            puck_high=(.3, .9),
            hand_low=(-0.15, 0.4, 0.05),
            hand_high=(0.15, .75, 0.3),
            norm_order=2,
            xml_path='sawyer_xyz/sawyer_push_puck_small_arena.xml',
            reward_type='state_distance',
            reset_free=False,
            clamp_puck_on_step=False,
        )
    )

    register(
        id='SawyerPushAndReachSmallArenaResetFreeEnv-v0',
        entry_point='multiworld.envs.mujoco.sawyer_xyz'
                    '.sawyer_push_and_reach_env:SawyerPushAndReachXYEnv',
        tags={
            'git-commit-hash': '7256aaf',
            'author': 'murtaza',
        },
        kwargs=dict(
            goal_low=(-0.15, 0.4, 0.02, -.1, .5),
            goal_high=(0.15, 0.75, 0.02, .1, .7),
            puck_low=(-.3, .25),
            puck_high=(.3, .9),
            hand_low=(-0.15, 0.4, 0.05),
            hand_high=(0.15, .75, 0.3),
            norm_order=2,
            xml_path='sawyer_xyz/sawyer_push_puck_small_arena.xml',
            reward_type='state_distance',
            reset_free=True,
            clamp_puck_on_step=False,
        )
    )

    """
    NIPS submission pusher environment
    """
    register(
        id='SawyerPushNIPS-v0',
        entry_point='multiworld.envs.mujoco.sawyer_xyz'
                    '.sawyer_push_nips:SawyerPushAndReachXYEasyEnv',
        tags={
            'git-commit-hash': 'bede25d',
            'author': 'ashvin',
        },
        kwargs=dict(
            hide_goal=True,
            reward_info=dict(
                type="state_distance",
            ),
        )

    )

    register(
        id='SawyerPushNIPSHarder-v0',
        entry_point='multiworld.envs.mujoco.sawyer_xyz'
                    '.sawyer_push_nips:SawyerPushAndReachXYHarderEnv',
        tags={
            'git-commit-hash': 'b5cac93',
            'author': 'murtaza',
        },
        kwargs=dict(
            hide_goal=True,
            reward_info=dict(
                type="state_distance",
            ),
        )

    )

    """
    Door Hook Env
    """

    register(
        id='SawyerDoorHookEnv-v0',
        entry_point='multiworld.envs.mujoco.sawyer_xyz'
                    '.sawyer_door_hook:SawyerDoorHookEnv',
        tags={
            'git-commit-hash': '15b48d5',
            'author': 'murtaza',
        },
        kwargs = dict(
            goal_low=(-0.1, 0.45, 0.1, 0),
            goal_high=(0.05, 0.65, .25, .83),
            hand_low=(-0.1, 0.45, 0.1),
            hand_high=(0.05, 0.65, .25),
            max_angle=.83,
            xml_path='sawyer_xyz/sawyer_door_pull_hook.xml',
            reward_type='angle_diff_and_hand_distance',
            reset_free=False,
        )
    )

    register(
        id='SawyerDoorHookResetFreeEnv-v0',
        entry_point='multiworld.envs.mujoco.sawyer_xyz'
                    '.sawyer_door_hook:SawyerDoorHookEnv',
        tags={
            'git-commit-hash': '15b48d5',
            'author': 'murtaza',
        },
        kwargs=dict(
            goal_low=(-0.1, 0.45, 0.1, 0),
            goal_high=(0.05, 0.65, .25, .83),
            hand_low=(-0.1, 0.45, 0.1),
            hand_high=(0.05, 0.65, .25),
            max_angle=.83,
            xml_path='sawyer_xyz/sawyer_door_pull_hook.xml',
            reward_type='angle_diff_and_hand_distance',
            reset_free=True,
        )
    )

    """
    Pick and Place
    """
    register(
        id='SawyerPickupEnv-v0',
        entry_point='multiworld.envs.mujoco.sawyer_xyz'
                    '.sawyer_pick_and_place:SawyerPickAndPlaceEnv',
        tags={
            'git-commit-hash': '30f23f7',
            'author': 'steven',
        },
        kwargs=dict(
            hand_low=(-0.1, 0.55, 0.05),
            hand_high=(0.0, 0.65, 0.2),
            action_scale=0.02,
            hide_goal_markers=True,
            num_goals_presampled=1000,
        )

    )
    register(
        id='SawyerPickupResetFreeEnv-v0',
        entry_point='multiworld.envs.mujoco.sawyer_xyz'
                    '.sawyer_pick_and_place:SawyerPickAndPlaceEnv',
        tags={
            'git-commit-hash': '30f23f7',
            'author': 'steven',
        },
        kwargs=dict(
            hand_low=(-0.1, 0.55, 0.05),
            hand_high=(0.0, 0.65, 0.2),
            action_scale=0.02,
            reset_free=True,
            hide_goal_markers=True,
            num_goals_presampled=1000,
        )

    )
    register(
        id='SawyerPickupEnvYZ-v0',
        entry_point='multiworld.envs.mujoco.sawyer_xyz'
                    '.sawyer_pick_and_place:SawyerPickAndPlaceEnvYZ',
        tags={
            'git-commit-hash': '30f23f7',
            'author': 'steven',
        },
        kwargs=dict(
            hand_low=(-0.1, 0.55, 0.05),
            hand_high=(0.0, 0.65, 0.2),
            action_scale=0.02,
            hide_goal_markers=True,
            num_goals_presampled=1000,
        )
    )


    register(
        id='SawyerPickupTallEnv-v0',
        entry_point='multiworld.envs.mujoco.sawyer_xyz'
                    '.sawyer_pick_and_place:SawyerPickAndPlaceEnv',
        tags={
            'git-commit-hash': '30f23f7',
            'author': 'steven',
        },
        kwargs=dict(
            hand_low=(-0.1, 0.55, 0.05),
            hand_high=(0.0, 0.65, 0.3),
            action_scale=0.02,
            hide_goal_markers=True,
            num_goals_presampled=1000,
        )

    )
    register(
        id='SawyerPickupWideEnv-v0',
        entry_point='multiworld.envs.mujoco.sawyer_xyz'
                    '.sawyer_pick_and_place:SawyerPickAndPlaceEnv',
        tags={
            'git-commit-hash': '30f23f7',
            'author': 'steven',
        },
        kwargs=dict(
            hand_low=(-0.2, 0.55, 0.05),
            hand_high=(0.0, 0.65, 0.2),
            action_scale=0.02,
            hide_goal_markers=True,
            num_goals_presampled=1000,
        )

    )

    register(
        id='SawyerPickupWideResetFreeEnv-v0',
        entry_point='multiworld.envs.mujoco.sawyer_xyz'
                    '.sawyer_pick_and_place:SawyerPickAndPlaceEnv',
        tags={
            'git-commit-hash': '30f23f7',
            'author': 'steven',
        },
        kwargs=dict(
            hand_low=(-0.2, 0.55, 0.05),
            hand_high=(0.0, 0.65, 0.2),
            action_scale=0.02,
            hide_goal_markers=True,
            reset_free=True,
            num_goals_presampled=1000,
        )

    )


    register(
        id='SawyerPickupTallWideEnv-v0',
        entry_point='multiworld.envs.mujoco.sawyer_xyz'
                    '.sawyer_pick_and_place:SawyerPickAndPlaceEnv',
        tags={
            'git-commit-hash': '30f23f7',
            'author': 'steven',
        },
        kwargs=dict(
            hand_low=(-0.2, 0.55, 0.05),
            hand_high=(0.0, 0.65, 0.3),
            action_scale=0.02,
            hide_goal_markers=True,
            num_goals_presampled=1000,
        )

    )

    """
    ICML Envs
    """
    register(
        id='SawyerPickupEnvYZEasy-v0',
        entry_point='multiworld.envs.mujoco.sawyer_xyz'
                    '.sawyer_pick_and_place:SawyerPickAndPlaceEnvYZ',
        tags={
            'git-commit-hash': '8bfd74b40f983e15026981344323b8e9539b4b21',
            'author': 'steven',
        },
        kwargs=dict(
            hand_low=(-0.1, 0.55, 0.05),
            hand_high=(0.0, 0.65, 0.13),
            action_scale=0.02,
            hide_goal_markers=True,
            num_goals_presampled=1000,

            p_obj_in_hand=.75,
        )
    )
    # This env is used for the image pickup version. We don't need state goals,
    # as the image env already generates image goals + state goals.
    register(
        id='SawyerPickupEnvYZEasyFewGoals-v0',
        entry_point='multiworld.envs.mujoco.sawyer_xyz'
                    '.sawyer_pick_and_place:SawyerPickAndPlaceEnvYZ',
        tags={
            'git-commit-hash': '8bfd74b40f983e15026981344323b8e9539b4b21',
            'author': 'steven',
        },
        kwargs=dict(
            hand_low=(-0.1, 0.55, 0.05),
            hand_high=(0.0, 0.65, 0.13),
            action_scale=0.02,
            hide_goal_markers=True,
            num_goals_presampled=1,

            p_obj_in_hand=.75,
        )
    )

    register(
        id='SawyerPickupEnvYZEasyImage48-v0',
        entry_point=create_image_48_sawyer_pickup_easy_v0,
        tags={
            'git-commit-hash': '8bfd74b40f983e15026981344323b8e9539b4b21',
            'author': 'steven'
        },
    )
    register(
        id='SawyerDoorHookResetFreeEnvImage48-v1',
        entry_point=create_image_48_sawyer_door_hook_reset_free_v1,
        tags={
            'git-commit-hash': '8bfd74b40f983e15026981344323b8e9539b4b21',
            'author': 'steven'
        },
    )
    register(
        id='SawyerPushNIPSEasy-v0',
        entry_point='multiworld.envs.mujoco.sawyer_xyz'
                    '.sawyer_push_nips:SawyerPushAndReachXYEasyEnv',
        tags={
            'git-commit-hash': 'b8d77fef5f3ebe4c1c9c3874a5e3faaab457a350',
            'author': 'steven',
        },
        kwargs=dict(
            force_puck_in_goal_space=False,
            mocap_low=(-0.1, 0.55, 0.0),
            mocap_high=(0.1, 0.65, 0.5),
            hand_goal_low=(-0.1, 0.55),
            hand_goal_high=(0.1, 0.65),
            puck_goal_low=(-0.15, 0.5),
            puck_goal_high=(0.15, 0.7),

            hide_goal=True,
            reward_info=dict(
                type="state_distance",
            ),
        )
    )
    register(
        id='SawyerPushNIPSEasyImage48-v0',
        entry_point='multiworld.core.image_env:ImageEnv',
        tags={
            'git-commit-hash': 'b8d77fef5f3ebe4c1c9c3874a5e3faaab457a350',
            'author': 'steven',
        },
        kwargs=dict(
            wrapped_env=gym.make('SawyerPushNIPSEasy-v0'),
            imsize=48,
            init_camera=sawyer_init_camera_zoomed_in,
            transpose=True,
            normalize=True,
        )
    )
    register(
        id='SawyerDoorHookResetFreeEnv-v1',
        entry_point='multiworld.envs.mujoco.sawyer_xyz'
                    '.sawyer_door_hook:SawyerDoorHookEnv',
        tags={
            'git-commit-hash': 'b8d77fef5f3ebe4c1c9c3874a5e3faaab457a350',
            'author': 'steven',
        },
        kwargs=dict(
            goal_low=(-0.1, 0.45, 0.1, 0),
            goal_high=(0.05, 0.65, .25, .83),
            hand_low=(-0.1, 0.45, 0.1),
            hand_high=(0.05, 0.65, .25),
            max_angle=.83,
            xml_path='sawyer_xyz/sawyer_door_pull_hook.xml',
            reward_type='angle_diff_and_hand_distance',
            reset_free=True,
        )
    )

    register(
        id='SawyerReachXYZEnv-v2',
        entry_point='multiworld.envs.mujoco.sawyer_xyz.sawyer_reach:SawyerReachXYZEnv',
        tags={
            'git-commit-hash': 'b8d77fef5f3ebe4c1c9c3874a5e3faaab457a350',
            'author': 'steven'
        },
        kwargs={
            'hide_goal_markers': True,
            'norm_order': 2,
        },
    )


def register_classic_mujoco_envs():
    register(
        id='LowGearAnt-v0',
        entry_point='multiworld.envs.mujoco.classic_mujoco.ant:AntEnv',
        kwargs={
            'use_low_gear_ratio': True,
        },
    )
    register(
        id='AntXY-v0',
        entry_point='multiworld.envs.mujoco.classic_mujoco.ant:AntXYGoalEnv',
        kwargs={
            'use_low_gear_ratio': False,
            'include_contact_forces_in_state': True
        },
    )
    register(
        id='AntXY-NoContactSensors-v0',
        entry_point='multiworld.envs.mujoco.classic_mujoco.ant:AntXYGoalEnv',
        kwargs={
            'use_low_gear_ratio': False,
            'include_contact_forces_in_state': False
        },
    )
    register(
        id='AntXY-LowGear-v0',
        entry_point='multiworld.envs.mujoco.classic_mujoco.ant:AntXYGoalEnv',
        kwargs={
            'use_low_gear_ratio': True,
            'include_contact_forces_in_state': True
        },
    )
    register(
        id='AntXY-LowGear-NoContactSensors-v0',
        entry_point='multiworld.envs.mujoco.classic_mujoco.ant:AntXYGoalEnv',
        kwargs={
            'use_low_gear_ratio': True,
            'include_contact_forces_in_state': False
        },
    )
    register(
        id='AntFullPositionGoal-v0',
        entry_point='multiworld.envs.mujoco.classic_mujoco.ant:AntFullPositionGoalEnv',
        kwargs={},
    )


def create_image_48_sawyer_reach_xy_env_v1():
    from multiworld.core.image_env import ImageEnv
    from multiworld.envs.mujoco.cameras import sawyer_xyz_reacher_camera_v0

    wrapped_env = gym.make('SawyerReachXYEnv-v1')
    return ImageEnv(
        wrapped_env,
        48,
        init_camera=sawyer_xyz_reacher_camera_v0,
        transpose=True,
        normalize=True,
    )


def create_image_84_sawyer_reach_xy_env_v1():
    from multiworld.core.image_env import ImageEnv
    from multiworld.envs.mujoco.cameras import sawyer_xyz_reacher_camera_v0

    wrapped_env = gym.make('SawyerReachXYEnv-v1')
    return ImageEnv(
        wrapped_env,
        84,
        init_camera=sawyer_xyz_reacher_camera_v0,
        transpose=True,
        normalize=True,
    )

def create_image_48_sawyer_push_and_reach_arena_env_v0():
    from multiworld.core.image_env import ImageEnv
    from multiworld.envs.mujoco.cameras import sawyer_pusher_camera_upright_v2

    wrapped_env = gym.make('SawyerPushAndReachArenaEnv-v0')
    return ImageEnv(
        wrapped_env,
        48,
        init_camera=sawyer_pusher_camera_upright_v2,
        transpose=True,
        normalize=True,
    )

def create_image_48_sawyer_push_and_reach_arena_env_reset_free_v0():
    from multiworld.core.image_env import ImageEnv
    from multiworld.envs.mujoco.cameras import sawyer_pusher_camera_upright_v2

    wrapped_env = gym.make('SawyerPushAndReachArenaResetFreeEnv-v0')
    return ImageEnv(
        wrapped_env,
        48,
        init_camera=sawyer_pusher_camera_upright_v2,
        transpose=True,
        normalize=True,
    )

def create_image_48_sawyer_door_hook_reset_free_v1():
    from multiworld.core.image_env import ImageEnv
    from multiworld.envs.mujoco.cameras import sawyer_door_env_camera_v0
    import os.path
    import numpy as np
    goal_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        'goals/door_goals.npy'
    )
    goals = np.load(goal_path).item()
    return ImageEnv(
        wrapped_env=gym.make('SawyerDoorHookResetFreeEnv-v1'),
        imsize=48,
        init_camera=sawyer_door_env_camera_v0,
        transpose=True,
        normalize=True,
        presampled_goals=goals,
    )

def create_image_48_sawyer_pickup_easy_v0():
    from multiworld.core.image_env import ImageEnv
    from multiworld.envs.mujoco.cameras import sawyer_pick_and_place_camera
    import os.path
    import numpy as np
    goal_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        'goals/pickup_goals.npy'
    )
    goals = np.load(goal_path).item()
    return ImageEnv(
        wrapped_env=gym.make('SawyerPickupEnvYZEasyFewGoals-v0'),
        imsize=48,
        init_camera=sawyer_pick_and_place_camera,
        transpose=True,
        normalize=True,
        presampled_goals=goals,
    )
