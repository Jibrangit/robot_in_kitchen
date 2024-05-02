"""controller that performs kitchen service"""

import sys
import os
import yaml

from py_trees.common import Status

root_dir = os.getenv("HOME") + "/webots/robot_planning"
sys.path.append(root_dir)

import numpy as np

from controller import Supervisor
from libraries.robot_device_io import RobotDeviceIO
from libraries.transformations import *


def convert_to_2d_matrix(array: list):
    return np.array([array[:4], array[4:8], array[8:12], array[12:]])


def main():
    robot = Supervisor()
    timestep = int(robot.getBasicTimeStep())

    # Supply all device handles to communicate with robot and all devices attached to it.
    robot_handle = RobotDeviceIO(robot)
    robot_handle.initialize_devices(timestep)
    robot_handle.set_motors_vels(0, 0)

    robot_body_handle = robot.getFromDef("TIAGO_ROBOT")
    torso_handle = robot.getFromDef("TORSO")
    torso_lift_handle = robot.getFromDef("TORSO_LIFT")
    arm_1_handle = robot.getFromDef("ARM_1")
    arm_2_handle = robot.getFromDef("ARM_2")

    frames = {
        ("TIAGO_ROBOT", "TORSO"): axis_angle_and_position_to_transformation_matrix(
            [0, 0, 1, 0], [-0.054, 0, 0.193]
        ),
        ("TORSO", "TORSO_LIFT_JOINT"): axis_angle_and_position_to_transformation_matrix(
            [0, 0, 1, 0], [0, 0, 0.6]
        ),
        (
            "TORSO_LIFT",
            "ARM_FRONT_EXTENSION",
        ): axis_angle_and_position_to_transformation_matrix(
            [0, 0, 1, -1.5708], [-0.037, 0.0388, 0.0224]
        ),
        (
            "ARM_FRONT_EXTENSION",
            "ARM_1_JOINT",
        ): axis_angle_and_position_to_transformation_matrix(
            [0, 0, 1, 0.00996], [0.0251, 0.194, -0.171]
        ),
        (
            "ARM_1",
            "ARM_2_JOINT",
        ): axis_angle_and_position_to_transformation_matrix(
            [1, 0, 0, np.pi / 2], [0.125, 0.018, -0.0311]
        ),
    }

    while robot.step(timestep) != -1:

        robot_pose = convert_to_2d_matrix(robot_body_handle.getPose())
        torso_pose = convert_to_2d_matrix(torso_handle.getPose())
        torso_lift_pose = convert_to_2d_matrix(torso_lift_handle.getPose())
        arm1_pose = convert_to_2d_matrix(arm_1_handle.getPose())
        arm2_pose = convert_to_2d_matrix(arm_2_handle.getPose())

        joint_positions = robot_handle.get_joint_positions()

        torso_pose_kinematics = robot_pose @ frames[("TIAGO_ROBOT", "TORSO")]
        torso_lift_pose_kinematics = (
            torso_pose_kinematics
            @ frames[("TORSO", "TORSO_LIFT_JOINT")]
            @ get_transformation_matrix(
                theta=0, d=joint_positions["torso_lift_joint"], a=0, alpha=0
            )
        )
        arm1_pose_kinematics = (
            torso_lift_pose_kinematics
            @ frames[("TORSO_LIFT", "ARM_FRONT_EXTENSION")]
            @ frames[
                (
                    "ARM_FRONT_EXTENSION",
                    "ARM_1_JOINT",
                )
            ]
            @ get_transformation_matrix(
                theta=joint_positions["arm_1_joint"] - 0.07, d=0, a=0, alpha=0
            )
        )

        arm2_pose_kinematics = (
            arm1_pose_kinematics
            @ frames[
                (
                    "ARM_1",
                    "ARM_2_JOINT",
                )
            ]
            @ get_transformation_matrix(
                theta=joint_positions["arm_2_joint"], d=0, a=0, alpha=0
            )
        )

        print(f"Arm2 pose from kinematics : {arm2_pose_kinematics}")
        print(f"Arm2 pose ground truth : {arm2_pose}")

        print("=============================================================")
        position_error = arm2_pose[:3, 3] - arm2_pose_kinematics[:3, 3]
        orientation_error = rotation_matrix_to_rot_vec(
            arm2_pose[:3, :3]
        ) - rotation_matrix_to_rot_vec(arm2_pose_kinematics[:3, :3])
        print(
            f"Position error = {np.sqrt(position_error @ np.transpose(position_error))}"
        )
        print(f"Orientation_error = {orientation_error}")


if __name__ == "__main__":
    main()
