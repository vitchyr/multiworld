from multiworld.envs.mujoco.sawyer_xyz.sawyer_pick_and_place import (
    SawyerPickAndPlaceEnv,
    SawyerPickAndPlaceEnvYZ,
)
from multiworld.core.image_env import ImageEnv
from multiworld.envs.mujoco.cameras import sawyer_pick_and_place_camera
from multiworld.envs.mujoco.util.image_util import render_cv2
import numpy as np

# env = SawyerPickAndPlaceEnvYZ(
    # hand_low=(-0.1, 0.55, 0.05),
    # hand_high=(0.0, 0.65, 0.15),
    # hand_reset_pos=(0.0, .55, .07),
    # action_scale=0.02,
    # hide_goal_markers=True,
    # num_goals_presampled=10,

    # p_obj_in_hand=.75
# )
import gym
from gym.envs import registration
import multiworld.envs.mujoco
env = gym.make('SawyerDoorHookResetFreeEnv-v0')

while True:

    action = env.action_space.sample()
    obs, _, _, _ = env.step(action)
    # obs, _, _, _ = image_env.step(np.zeros(3))
    env.render()
    # render_cv2(obs)
