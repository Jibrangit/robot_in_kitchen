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


def print_error(pose, pose_kinematics, name: str):
    print(f"{name} pose from kinematics : {pose_kinematics}")
    print(f"{name} pose ground truth : {pose}")

    print("=============================================================")
    position_error = pose[:3, 3] - pose_kinematics[:3, 3]
    orientation_error = rotation_matrix_to_rot_vec(
        pose[:3, :3]
    ) - rotation_matrix_to_rot_vec(pose_kinematics[:3, :3])
    print(f"Position error = {np.sqrt(position_error @ np.transpose(position_error))}")
    print(f"Orientation_error = {orientation_error}")


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
    arm_3_handle = robot.getFromDef("ARM_3")

    TORSO_LIFT_INITIAL_POSITION = 0.6
    ARM_1_INITIAL_POSITION = 0.07
    ARM_2_INITIAL_POSITION = np.pi / 2

    frames = {
        ("TIAGO_ROBOT", "TORSO"): axis_angle_and_position_to_transformation_matrix(
            [0, 0, 1, 0], [-0.054, 0, 0.193]
        ),
        ("TORSO", "TORSO_LIFT_JOINT"): axis_angle_and_position_to_transformation_matrix(
            [0, 0, 1, 0], [0, 0, TORSO_LIFT_INITIAL_POSITION]
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
            [1, 0, 0, 0], [0.125, 0.018, -0.0311]
        ),
    }

    while robot.step(timestep) != -1:

        robot_pose = convert_to_2d_matrix(robot_body_handle.getPose())
        torso_pose = convert_to_2d_matrix(torso_handle.getPose())
        torso_lift_pose = convert_to_2d_matrix(torso_lift_handle.getPose())
        arm1_pose = convert_to_2d_matrix(arm_1_handle.getPose())
        arm2_pose = convert_to_2d_matrix(arm_2_handle.getPose())
        arm3_pose = convert_to_2d_matrix(arm_3_handle.getPose())

        joint_positions = robot_handle.get_joint_positions()

        torso_pose_kinematics = robot_pose @ frames[("TIAGO_ROBOT", "TORSO")]
        torso_lift_pose_kinematics = (
            torso_pose_kinematics
            @ frames[("TORSO", "TORSO_LIFT_JOINT")]
            @ get_translation_matrix(
                tx=0,
                ty=0,
                tz=joint_positions["torso_lift_joint"],
            )
        )

        arm_1_pose_kinematics = (
            torso_lift_pose_kinematics
            @ frames[
                (
                    "TORSO_LIFT",
                    "ARM_FRONT_EXTENSION",
                )
            ]
            @ frames[
                (
                    "ARM_FRONT_EXTENSION",
                    "ARM_1_JOINT",
                )
            ]
            @ transformation_matrix_from_rot_axis_and_translation(
                rot_angle=joint_positions["arm_1_joint"] - ARM_1_INITIAL_POSITION,
                rot_axis="z",
            )
        )

        arm_2_pose_kinematics = (
            arm_1_pose_kinematics
            @ transformation_matrix_from_rot_axis_and_translation(
                rot_angle=-joint_positions["arm_2_joint"],
                rot_axis="y",
                trans_vec=[0.125, 0, -0.0015]
            )
            @ axis_angle_and_position_to_transformation_matrix(
                axis_angle_vector=[1, 0, 0, np.pi / 2], positions=[0, 0, 0]
            )
        )

        arm_3_pose_kinematics = (
            arm_2_pose_kinematics
            @ transformation_matrix_from_rot_axis_and_translation(
                rot_angle=-joint_positions["arm_3_joint"],
                rot_axis="x",
                trans_vec=[0.0872, 0, -0.0015]
            )
            @ axis_angle_and_position_to_transformation_matrix(
                axis_angle_vector=[0, 0.707, 0.707, np.pi], positions=[0, 0, 0]
            )
        )

        print_error(arm3_pose, arm_3_pose_kinematics, "arm_3")


if __name__ == "__main__":
    main()
