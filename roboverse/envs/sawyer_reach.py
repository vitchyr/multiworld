from roboverse.envs.sawyer_grasp import SawyerGraspOneEnv


class SawyerReachEnv(SawyerGraspOneEnv):

    def get_reward(self, info):

        if self._reward_type == 'sparse':
            if info['object_gripper_distance'] < 0.03:
                reward = 1
            else:
                reward = 0
        elif self._reward_type == 'shaped':
            reward = -1*info['object_gripper_distance']
            reward = max(reward, self._reward_min)
        else:
            raise NotImplementedError

        return reward
