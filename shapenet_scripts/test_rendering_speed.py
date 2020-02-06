import time
import roboverse


def main():
    # env = roboverse.make('SawyerGraspOne-v0', observation_mode='pixels', gui=True)
    env = roboverse.make('SawyerGraspOne-v0', observation_mode='pixels', gui=False)
    start_time = time.time()

    for j in range(3):
        env.reset()
        for i in range(50):
            env.step(env.action_space.sample())

    print("--- %s seconds ---" % (time.time() - start_time))

if __name__ == '__main__':
    main()