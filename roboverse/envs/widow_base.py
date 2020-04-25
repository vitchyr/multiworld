import numpy as np
import gym
import pdb

import roboverse.bullet as bullet
from roboverse.envs.robot_base import RobotBaseEnv


class WidowBaseEnv(RobotBaseEnv):
    def __init__(self, *args, **kwargs):

        self._id = 'WidowBaseEnv'
        self._robot_name = 'widowx'
        self._gripper_joint_name = ('gripper_prismatic_joint_1', 'gripper_prismatic_joint_2')
        if self._env_name == 'WidowX200GraspEnv':
            self._gripper_joint_name = ('left_finger', 'right_finger')

        self._gripper_range = range(7, 9)
        self.downwards = kwargs.get('downwards')
        del kwargs['downwards']

        self._end_effector_link_name = 'gripper_rail_link'
        
        if self._env_name == 'WidowX200GraspEnv':
            self._end_effector_link_name = 'wx200/gripper_bar_link' 
        
        super().__init__(*args, **kwargs)
        self._load_meshes()
        self._end_effector = self._end_effector = bullet.get_index_by_attribute(
            self._robot_id, 'link_name', self._end_effector_link_name)

        self._setup_environment()

    def _load_meshes(self):
        if self.downwards:
            self._robot_id = bullet.objects.widow_downwards()
        else:
            print("self._env_name", self._env_name)
            if self._env_name in ['WidowX200GraspEnv', 'Widow200GraspV2Env']:
                self._robot_id = bullet.objects.widowx_200()
            else:
                self._robot_id = bullet.objects.widow()
        self._table = bullet.objects.table()
        self._objects = {}
        self._workspace = bullet.Sensor(self._robot_id,
                                        xyz_min=self._pos_low, xyz_max=self._pos_high,
                                        visualize=False, rgba=[0, 1, 0, .1])

