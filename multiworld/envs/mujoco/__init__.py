import gym
from gym.envs.registration import register
import logging

LOGGER = logging.getLogger(__name__)

_REGISTERED = False


def register_custom_envs():
    global _REGISTERED
    if _REGISTERED:
        return
    _REGISTERED = True

    LOGGER.info("Registering multiworld mujoco gym environments")

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
        id='Image48SawyerReachXYEnv-v1',
        entry_point=create_image_48_sawyer_reach_xy_env_v1,
        tags={
            'git-commit-hash': '2d95c75',
            'author': 'murtaza'
        },
    )

    register(
        id='SawyerReachXYEnv-v2',
        entry_point='multiworld.envs.mujoco.sawyer_xyz.sawyer_reach:SawyerReachXYEnv',
        tags={
            'git-commit-hash': '2d95c75', #9cab5da
            'author': 'Soroush'
        },
        kwargs={
            'reward_type': 'vectorized_hand_distance',
            'norm_order': 2,
            'hide_goal_markers': True,
        }
    )
    register(
        id='Image84SawyerReachXYEnv-v2',
        entry_point=create_image_84_sawyer_reach_xy_env_v2,
        tags={
            'git-commit-hash': '2d95c75', #9cab5da
            'author': 'Soroush'
        },
    )

    register(
        id='SawyerReachXYEnv-v3',
        entry_point='multiworld.envs.mujoco.sawyer_xyz.sawyer_reach:SawyerReachXYEnv',
        tags={
            'git-commit-hash': '887848a',
            'author': 'Soroush'
        },
        kwargs={
            'reward_type': 'vectorized_hand_distance',
            'norm_order': 2,
            'hide_goal_markers': True,
            'fix_reset': False,
            'action_scale': 0.01,
        }
    )
    register(
        id='Image84SawyerReachXYEnv-v3',
        entry_point=create_image_84_sawyer_reach_xy_env_v3,
        tags={
            'git-commit-hash': '887848a',
            'author': 'Soroush'
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

    """
    Pick and Place
    """
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
            num_goals_presampled=30,
        )
    )


    """
    Wheeled Car
    """
    register(
        id='WheeledCarEnv-v0',
        entry_point='multiworld.envs.mujoco.locomotion.wheeled_car:WheeledCarEnv',
        tags={
            'git-commit-hash': 'f773062',
            'author': 'Soroush'
        },
        kwargs={
            'reward_type': 'vectorized_dense',
            'norm_order': 2,
            'car_low':(-1.60, -1.60),
            'car_high':(1.60, 1.60),
            'goal_low':(-1.60, -1.60),
            'goal_high':(1.60, 1.60),
        }
    )
    register(
        id='Image84WheeledCarEnv-v0',
        entry_point=create_image_84_wheeled_car_env_v0,
        tags={
            'git-commit-hash': 'f773062',
            'author': 'Soroush'
        },
    )

    register(
        id='WheeledCarEnv-v1',
        entry_point='multiworld.envs.mujoco.locomotion.wheeled_car:WheeledCarEnv',
        tags={
            'git-commit-hash': 'f773062',
            'author': 'Soroush'
        },
        kwargs={
            'reward_type': 'vectorized_dense',
            'norm_order': 2,
            'car_low':(-1.35, -1.35),
            'car_high':(1.35, 1.35),
            'goal_low':(-1.35, -1.35),
            'goal_high':(1.35, 1.35),
        }
    )
    register(
        id='Image84WheeledCarEnv-v1',
        entry_point=create_image_84_wheeled_car_env_v1,
        tags={
            'git-commit-hash': 'f773062',
            'author': 'Soroush'
        },
    )

    register_soroush_envs()

def create_image_84_wheeled_car_env_v0():
    from multiworld.core.image_env import ImageEnv
    from multiworld.envs.mujoco.cameras import wheeled_car_camera_v0

    wrapped_env = gym.make('WheeledCarEnv-v0')
    return ImageEnv(
        wrapped_env,
        84,
        init_camera=wheeled_car_camera_v0,
        transpose=True,
        normalize=True,
    )

def create_image_84_wheeled_car_env_v1():
    from multiworld.core.image_env import ImageEnv
    from multiworld.envs.mujoco.cameras import wheeled_car_camera_v0

    wrapped_env = gym.make('WheeledCarEnv-v1')
    return ImageEnv(
        wrapped_env,
        84,
        init_camera=wheeled_car_camera_v0,
        transpose=True,
        normalize=True,
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


def create_image_84_sawyer_reach_xy_env_v2():
    from multiworld.core.image_env import ImageEnv
    from multiworld.envs.mujoco.cameras import init_sawyer_camera_v1

    wrapped_env = gym.make('SawyerReachXYEnv-v2')
    return ImageEnv(
        wrapped_env,
        84,
        init_camera=init_sawyer_camera_v1,
        transpose=True,
        normalize=True,
    )

def create_image_84_sawyer_reach_xy_env_v3():
    from multiworld.core.image_env import ImageEnv
    from multiworld.envs.mujoco.cameras import init_sawyer_camera_v1

    wrapped_env = gym.make('SawyerReachXYEnv-v3')
    return ImageEnv(
        wrapped_env,
        84,
        init_camera=init_sawyer_camera_v1,
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

def register_soroush_envs():
    register(
        id='SawyerPushAndReachTrainEnvEasy-v0',
        entry_point='multiworld.envs.mujoco.sawyer_xyz'
                    '.sawyer_push_nips:SawyerPushAndReachXYEnv',
        tags={
            'git-commit-hash': '0f24d69',
            'author': 'Soroush',
        },
        kwargs=dict(
            hand_low=(-0.10, 0.55),
            hand_high=(0.10, 0.65),
            puck_low=(-0.10, 0.55),
            puck_high=(0.10, 0.65),
            fix_reset=0.075,
            sample_realistic_goals=True,
            reward_type='state_distance',
        )
    )
    register(
        id='SawyerPushAndReachTrainEnvEasyVectRew-v0',
        entry_point='multiworld.envs.mujoco.sawyer_xyz'
                    '.sawyer_push_nips:SawyerPushAndReachXYEnv',
        tags={
            'git-commit-hash': '0f24d69',
            'author': 'Soroush',
        },
        kwargs=dict(
            hand_low=(-0.10, 0.55),
            hand_high=(0.10, 0.65),
            puck_low=(-0.10, 0.55),
            puck_high=(0.10, 0.65),
            fix_reset=0.075,
            sample_realistic_goals=True,
            reward_type='vectorized_state_distance',
        )
    )
    register(
        id='Image84SawyerPushAndReachTrainEnvEasy-v0',
        entry_point=create_image_84_sawyer_pnr_train_env_easy_v0,
        tags={
            'git-commit-hash': '91f090c',
            'author': 'Soroush'
        },
    )

    register(
        id='SawyerPushAndReachTrainEnvHard-v0',
        entry_point='multiworld.envs.mujoco.sawyer_xyz'
                    '.sawyer_push_nips:SawyerPushAndReachXYEnv',
        tags={
            'git-commit-hash': '91f090c',
            'author': 'Soroush',
        },
        kwargs=dict(
            hand_low=(-0.20, 0.50),
            hand_high=(0.20, 0.70),
            puck_low=(-0.20, 0.50),
            puck_high=(0.20, 0.70),
            fix_reset=0.075,
            sample_realistic_goals=True,
            reward_type='state_distance',
        )
    )
    register(
        id='SawyerPushAndReachTrainEnvHardVectRew-v0',
        entry_point='multiworld.envs.mujoco.sawyer_xyz'
                    '.sawyer_push_nips:SawyerPushAndReachXYEnv',
        tags={
            'git-commit-hash': '91f090c',
            'author': 'Soroush',
        },
        kwargs=dict(
            hand_low=(-0.20, 0.50),
            hand_high=(0.20, 0.70),
            puck_low=(-0.20, 0.50),
            puck_high=(0.20, 0.70),
            fix_reset=0.075,
            sample_realistic_goals=True,
            reward_type='vectorized_state_distance',
        )
    )

    register(
        id='SawyerPushAndReachTestEnvHard-v0',
        entry_point='multiworld.envs.mujoco.sawyer_xyz'
                    '.sawyer_push_nips:SawyerPushAndReachXYEnv',
        tags={
            'git-commit-hash': '91f090c',
            'author': 'Soroush',
        },
        kwargs=dict(
            hand_low=(-0.20, 0.50),
            hand_high=(0.20, 0.70),
            puck_low=(-0.20, 0.50),
            puck_high=(0.20, 0.70),
            reset_low=(0.15, 0.65, -0.15, 0.55),
            reset_high=(0.20, 0.70, -0.10, 0.60),
            goal_low=(-0.20, 0.50, 0.15, 0.65),
            goal_high=(-0.15, 0.55, 0.20, 0.70),
            fix_reset=False,
            sample_realistic_goals=True,
            reward_type='state_distance',
        )
    )
    register(
        id='SawyerPushAndReachTestEnvHardVectRew-v0',
        entry_point='multiworld.envs.mujoco.sawyer_xyz'
                    '.sawyer_push_nips:SawyerPushAndReachXYEnv',
        tags={
            'git-commit-hash': '91f090c',
            'author': 'Soroush',
        },
        kwargs=dict(
            hand_low=(-0.20, 0.50),
            hand_high=(0.20, 0.70),
            puck_low=(-0.20, 0.50),
            puck_high=(0.20, 0.70),
            reset_low=(0.15, 0.65, -0.15, 0.55),
            reset_high=(0.20, 0.70, -0.10, 0.60),
            goal_low=(-0.20, 0.50, 0.15, 0.65),
            goal_high=(-0.15, 0.55, 0.20, 0.70),
            fix_reset=False,
            sample_realistic_goals=True,
            reward_type='vectorized_state_distance',
        )
    )

    register(
        id='Image84SawyerPushAndReachTrainEnvHard-v0',
        entry_point=create_image_84_sawyer_pnr_train_env_hard_v0,
        tags={
            'git-commit-hash': '91f090c',
            'author': 'Soroush'
        },
    )
    register(
        id='Image84SawyerPushAndReachTestEnvHard-v0',
        entry_point=create_image_84_sawyer_pnr_test_env_hard_v0,
        tags={
            'git-commit-hash': '91f090c',
            'author': 'Soroush'
        },
    )

def create_image_84_sawyer_pnr_train_env_easy_v0():
    from multiworld.core.image_env import ImageEnv
    from multiworld.envs.mujoco.cameras import sawyer_pusher_camera_tdm

    wrapped_env = gym.make('SawyerPushAndReachTrainEnvEasy-v0')
    return ImageEnv(
        wrapped_env,
        84,
        init_camera=sawyer_pusher_camera_tdm,
        transpose=True,
        normalize=True,
    )
def create_image_84_sawyer_pnr_train_env_hard_v0():
    from multiworld.core.image_env import ImageEnv
    from multiworld.envs.mujoco.cameras import sawyer_pusher_camera_tdm_v4

    wrapped_env = gym.make('SawyerPushAndReachTrainEnvHard-v0')
    return ImageEnv(
        wrapped_env,
        84,
        init_camera=sawyer_pusher_camera_tdm_v4,
        transpose=True,
        normalize=True,
    )
def create_image_84_sawyer_pnr_test_env_hard_v0():
    from multiworld.core.image_env import ImageEnv
    from multiworld.envs.mujoco.cameras import sawyer_pusher_camera_tdm_v4

    wrapped_env = gym.make('SawyerPushAndReachTestEnvHard-v0')
    return ImageEnv(
        wrapped_env,
        84,
        init_camera=sawyer_pusher_camera_tdm_v4,
        transpose=True,
        normalize=True,
    )

register_custom_envs()
