"""controller that performs kitchen service"""

import sys
import os
import yaml

root_dir = os.getenv("HOME") + "/webots/robot_planning"
sys.path.append(root_dir)

import numpy as np

from controller import Supervisor
from libraries.robot_device_io import RobotDeviceIO
from libraries.transformations import *
from libraries.tiago_kinematics import *


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
    arm_4_handle = robot.getFromDef("ARM_4")
    arm_5_handle = robot.getFromDef("ARM_5")
    arm_6_handle = robot.getFromDef("ARM_6")
    wrist_handle = robot.getFromDef("WRIST")
    left_gripper_handle = robot.getFromDef("LEFT_GRIPPER")
    right_gripper_handle = robot.getFromDef("RIGHT_GRIPPER")

    TORSO_LIFT_INITIAL_POSITION = 0.6
    ARM_1_INITIAL_POSITION = 0.07
    ARM_2_INITIAL_POSITION = np.pi / 2

    tiago_kinematics = TiagoKinematics()

    while robot.step(timestep) != -1:

        robot_pose = convert_to_2d_matrix(robot_body_handle.getPose())
        torso_pose = convert_to_2d_matrix(torso_handle.getPose())
        torso_lift_pose = convert_to_2d_matrix(torso_lift_handle.getPose())
        arm1_pose = convert_to_2d_matrix(arm_1_handle.getPose())
        arm2_pose = convert_to_2d_matrix(arm_2_handle.getPose())
        arm3_pose = convert_to_2d_matrix(arm_3_handle.getPose())
        arm4_pose = convert_to_2d_matrix(arm_4_handle.getPose())
        arm5_pose = convert_to_2d_matrix(arm_5_handle.getPose())
        arm6_pose = convert_to_2d_matrix(arm_6_handle.getPose())
        wrist_pose = convert_to_2d_matrix(wrist_handle.getPose())
        left_gripper_pose = convert_to_2d_matrix(left_gripper_handle.getPose())
        right_gripper_pose = convert_to_2d_matrix(right_gripper_handle.getPose())

        joint_positions = robot_handle.get_joint_positions()

        torso_pose_kinematics = robot_pose @ tiago_kinematics.transform_tree.get_pose("TIAGO_ROBOT", "TORSO")
        print_error(torso_pose, torso_pose_kinematics, "TORSO")


if __name__ == "__main__":
    main()
