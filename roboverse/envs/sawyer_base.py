import roboverse.bullet as bullet
from roboverse.envs.robot_base import RobotBaseEnv


class SawyerBaseEnv(RobotBaseEnv):
    def __init__(self,
                 img_dim=256,
                 gui=False,
                 action_scale=.2,
                 action_repeat=10,
                 timestep=1. / 120,
                 solver_iterations=150,
                 gripper_bounds=[-1, 1],
                 pos_init=[0.5, 0, 0],
                 pos_high=[1, .4, .25],
                 pos_low=[.4, -.6, -.36],
                 max_force=1000.,
                 visualize=True,
                 ):

        super().__init__(img_dim,
                         gui,
                         action_scale,
                         action_repeat,
                         timestep,
                         solver_iterations,
                         gripper_bounds,
                         pos_init,
                         pos_high,
                         pos_low,
                         max_force,
                         visualize)

        self._id = 'SawyerBaseEnv'
        self._robot_name = 'sawyer'
        self._gripper_joint_name = ('right_gripper_l_finger_joint', 'right_gripper_r_finger_joint')
        self._gripper_range = range(20, 25)

        self._r_limits = {}
        self._l_limits = {}

        self._load_meshes()
        self._end_effector = self._end_effector = bullet.get_index_by_attribute(
            self._robot_id, 'link_name', 'gripper_site')
        self._setup_environment()


    def _load_meshes(self):
        self._robot_id = bullet.objects.sawyer()
        self._table = bullet.objects.table()
        self._objects = {}
        self._sensors = {}
        self._workspace = bullet.Sensor(self._robot_id,
            xyz_min=self._pos_low, xyz_max=self._pos_high,
            visualize=False, rgba=[0,1,0,.1])
