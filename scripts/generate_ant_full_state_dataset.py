import numpy as np

from multiworld.envs.env_util import get_asset_full_path
from multiworld.envs.mujoco.classic_mujoco.ant import AntFullPositionGoalEnv

env = AntFullPositionGoalEnv()
def sample_start_xy():
    return np.random.uniform(-5, 5, size=2)

num_rollouts = 5000
num_steps_per_rollout = 30
save_every = 10
render = False
positions = []
velocities = []
for _ in range(num_rollouts):
    env.reset()
    start_xy = sample_start_xy()
    # import ipdb; ipdb.set_trace()
    state = env.sim.get_state()
    state.qpos[:2] = start_xy
    env.sim.set_state(state)
    # env.sim.model.geom_pos[1:, 0:2] += start_xy
    for t in range(num_steps_per_rollout):
        env.step(env.action_space.sample())
        if t % save_every == 0:
            state = env.sim.get_state()
            positions.append(state.qpos.copy()[:15])
            velocities.append(state.qvel.copy()[:14])
        if render:
            env.render()

print(len(positions))
positions = np.array(positions)
positions_path = get_asset_full_path('classic_mujoco/ant_goal_qpos_5x5_xy.npy')
np.save(positions_path, positions)

velocities = np.array(velocities)
velocities_path = get_asset_full_path('classic_mujoco/ant_goal_qvel_5x5_xy.npy')
np.save(velocities_path, positions)
