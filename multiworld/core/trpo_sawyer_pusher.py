
from sandbox.rocky.tf.algos.trpo import TRPO
from sandbox.rocky.tf.algos.vpg import VPG
from sandbox.rocky.tf.policies.minimal_gauss_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.mujoco.ant_env_dense import  AntEnvRandGoalRing
#rom rllab.envs.mujoco.blockpush_env_sparse import BlockPushEnv
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite

from multiworld.envs.mujoco.sawyer_xyz.push.sawyer_push import SawyerPushEnv


#stub(globals())
rate =0.01
mode = 'ec2'

import tensorflow as tf
for goal in range(1,100):
    env = normalize(AntEnvRandGoalRing())
    #env = WheeledEnvGoal()
   

    env = TfEnv(env)
    policy = GaussianMLPPolicy(
        name='policy',
        env_spec=env.spec,
        hidden_nonlinearity=tf.nn.relu,
        hidden_sizes=(100, 100)
    )

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=10000,
        max_path_length=200,
        n_itr=300,
        discount=0.99,
        step_size=rate,
        reset_arg = goal
        #plot=True,
    )

    run_experiment_lite(
        algo.train(),
        # Number of parallel workers for sampling
        n_parallel=4,
        # Only keep the snapshot parameters for the last iteration
        snapshot_mode="all",
        # Specifies the seed for the experiment. If this is not provided, a random seed
        # will be used
        seed=1,
        exp_prefix='TRPO_ant_dense',
        exp_name='Task' + str(goal),
        
        mode = mode,
        sync_s3_pkl = True
        #plot=True,
    )
