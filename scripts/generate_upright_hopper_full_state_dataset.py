import numpy as np

from multiworld.envs.env_util import get_asset_full_path
from multiworld.envs.mujoco.classic_mujoco.ant import AntFullPositionGoalEnv
from multiworld.envs.mujoco.classic_mujoco.hopper import \
    HopperFullPositionGoalEnv

env = HopperFullPositionGoalEnv()
def sample_start_x():
    return np.random.uniform(-10, 10, size=1)

num_rollouts = 5000
num_steps_per_rollout = 30
save_every = 10
render = False
positions = []
velocities = []
for _ in range(num_rollouts):
    env.reset()
    start_x = sample_start_x()
    # import ipdb; ipdb.set_trace()
    state = env.sim.get_state()
    state.qpos[:1] = start_x
    env.sim.set_state(state)
    # env.sim.model.geom_pos[1:, 0:2] += start_xy
    for t in range(num_steps_per_rollout):
        action = env.action_space.sample()
        action = 2 * np.random.beta(0.75, 0.75, action.shape) - 1
        # print(action)

        # action = np.ones_like(action)
        env.step(action)
        if t % save_every == 0 and t > 0:
            state = env.sim.get_state()
            positions.append(state.qpos.copy()[:6])
            velocities.append(state.qvel.copy()[:6])
        if render:
            env.render()

print(len(positions))
positions = np.array(positions)
positions_path = get_asset_full_path('classic_mujoco/hopper_goal_qpos_-10to10_x_upright.npy')
np.save(positions_path, positions)

velocities = np.array(velocities)
velocities_path = get_asset_full_path('classic_mujoco/hopper_goal_qvel_-10to10_x_upright.npy')
np.save(velocities_path, positions)
