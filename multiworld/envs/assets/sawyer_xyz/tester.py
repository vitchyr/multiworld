from multiworld.envs.mujoco.sawyer_xyz.sawyer_pick_and_place import SawyerPickAndPlaceEnv



env = SawyerPickAndPlaceEnv()
env.reset()

for i in range(int(1e6)):

    env.step()
    env.render()
    #env.step([100,100,100,100])
