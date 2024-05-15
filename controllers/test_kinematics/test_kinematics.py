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
from copy import deepcopy


def convert_to_2d_matrix(array: list):
    return np.array([array[:4], array[4:8], array[8:12], array[12:]])


def print_error(pose, pose_kinematics, name: str):
    print(f"{name} pose from kinematics : {pose_kinematics}")
    print(f"{name} pose ground truth : {pose}")

    position_error = pose[:3, 3] - pose_kinematics[:3, 3]
    orientation_error = rotation_matrix_to_rot_vec(
        pose[:3, :3]
    ) - rotation_matrix_to_rot_vec(pose_kinematics[:3, :3])
    print(f"Position error = {np.sqrt(position_error @ np.transpose(position_error))}")
    print(f"Orientation_error = {orientation_error}")
    print("=============================================================")


def clamp_error(error_vec):
    new_error_vec = error_vec.reshape(1, len(error_vec))[0]
    print(new_error_vec)
    D_max = 1e-2
    for idx, error in enumerate(new_error_vec):
        if error > 0:
            if error > D_max:
                new_error_vec[idx] = D_max

        elif error < 0:
            if error < -D_max:
                new_error_vec[idx] = -D_max

    return new_error_vec.reshape(len(error_vec), 1)

def get_error_mag(error_vec):
    error_vec = np.reshape(error_vec, (1, len(error_vec)))[0]
    mag = 0
    for error in error_vec:
        mag += error**2 
    return np.sqrt(mag)

def main():
    robot = Supervisor()
    timestep = int(robot.getBasicTimeStep())

    # Supply all device handles to communicate with robot and all devices attached to it.
    robot_handle = RobotDeviceIO(robot)
    robot_handle.initialize_devices(timestep)
    robot_handle.set_motors_vels(0, 0)
    # robot_handle.set_joint_positions(
    #     {
    #         "torso_lift_joint": np.inf,
    #         "arm_1_joint": np.inf,
    #         "arm_2_joint": np.inf,
    #         "arm_3_joint": np.inf,
    #         "arm_4_joint": np.inf,
    #         "arm_5_joint": np.inf,
    #         "arm_6_joint": np.inf,
    #         "arm_7_joint": np.inf,
    #     }
    # )

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

    tiago_kinematics = TiagoKinematics()
    robot_handle.set_joint_velocities({"arm_1_joint": 1.5})

    wrist_prev_position = [0, 0, 0]
    goal_wrist_position = None

    first = False

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

        wrist_position = wrist_pose[:3, -1]
        wrist_velocity = (wrist_position - wrist_prev_position) / (timestep * 0.001)
        # print(f"Wrist velocity = {wrist_velocity}")
        wrist_prev_position = wrist_position

        joint_positions = robot_handle.get_joint_positions()

        torso_pose_kinematics = robot_pose @ tiago_kinematics.transform_tree.get_pose(
            "TIAGO_ROBOT", "TORSO"
        )

        # Update transforms with joint positions
        tiago_kinematics.transform_tree.update_joint_controlled_transform(
            "TORSO_LIFT", joint_positions["torso_lift_joint"]
        )
        tiago_kinematics.transform_tree.update_joint_controlled_transform(
            "ARM_1", joint_positions["arm_1_joint"]
        )
        tiago_kinematics.transform_tree.update_joint_controlled_transform(
            "ARM_2", joint_positions["arm_2_joint"]
        )
        tiago_kinematics.transform_tree.update_joint_controlled_transform(
            "ARM_3", joint_positions["arm_3_joint"]
        )
        tiago_kinematics.transform_tree.update_joint_controlled_transform(
            "ARM_4", joint_positions["arm_4_joint"]
        )
        tiago_kinematics.transform_tree.update_joint_controlled_transform(
            "ARM_5", joint_positions["arm_5_joint"]
        )
        tiago_kinematics.transform_tree.update_joint_controlled_transform(
            "ARM_6", joint_positions["arm_6_joint"]
        )
        tiago_kinematics.transform_tree.update_joint_controlled_transform(
            "ARM_7", joint_positions["arm_7_joint"]
        )
        tiago_kinematics.transform_tree.update_joint_controlled_transform(
            "LEFT_GRIPPER", joint_positions["gripper_left_finger_joint"]
        )
        tiago_kinematics.transform_tree.update_joint_controlled_transform(
            "RIGHT_GRIPPER", joint_positions["gripper_right_finger_joint"]
        )

        torso_lift_pose_kinematics = (
            robot_pose
            @ tiago_kinematics.transform_tree.get_pose("TIAGO_ROBOT", "TORSO_LIFT")
        )
        arm1_pose_kinematics = robot_pose @ tiago_kinematics.transform_tree.get_pose(
            "TIAGO_ROBOT", "ARM_1"
        )
        arm2_pose_kinematics = robot_pose @ tiago_kinematics.transform_tree.get_pose(
            "TIAGO_ROBOT", "ARM_2"
        )
        arm3_pose_kinematics = robot_pose @ tiago_kinematics.transform_tree.get_pose(
            "TIAGO_ROBOT", "ARM_3"
        )
        arm4_pose_kinematics = robot_pose @ tiago_kinematics.transform_tree.get_pose(
            "TIAGO_ROBOT", "ARM_4"
        )
        arm5_pose_kinematics = robot_pose @ tiago_kinematics.transform_tree.get_pose(
            "TIAGO_ROBOT", "ARM_5"
        )
        arm6_pose_kinematics = robot_pose @ tiago_kinematics.transform_tree.get_pose(
            "TIAGO_ROBOT", "ARM_6"
        )
        arm7_pose_kinematics = robot_pose @ tiago_kinematics.transform_tree.get_pose(
            "TIAGO_ROBOT", "ARM_7"
        )
        wrist_pose_kinematics = robot_pose @ tiago_kinematics.transform_tree.get_pose(
            "TIAGO_ROBOT", "WRIST"
        )
        left_gripper_pose_kinematics = (
            robot_pose
            @ tiago_kinematics.transform_tree.get_pose("TIAGO_ROBOT", "LEFT_GRIPPER")
        )
        right_gripper_pose_kinematics = (
            robot_pose
            @ tiago_kinematics.transform_tree.get_pose("TIAGO_ROBOT", "RIGHT_GRIPPER")
        )

        wrist_pose_wrt_robot = tiago_kinematics.transform_tree.get_pose(
            "TIAGO_ROBOT", "WRIST"
        )
        wrist_position_kinematics = wrist_pose_wrt_robot[:3, -1]
        # robot_handle.joints_to_home_positions()

        # print_error(wrist_pose, wrist_pose_kinematics, "WRIST")

        jacobian, joint_list = tiago_kinematics.transform_tree.get_complete_jacobian(
            "WRIST"
        )
        # print(joint_list)

        if np.linalg.det(jacobian @ np.transpose(jacobian)) < 1e-2:
            print("Jacobian is singular!!!!!!!!!")

        linear_jacobian = jacobian[:3, :]
        # print(np.transpose(linear_jacobian))
        # print("===============================")

        if not first:
            goal_wrist_position = deepcopy(wrist_position_kinematics)
            goal_wrist_position[0] += 0.1
            goal_wrist_position[1] += 0.3
            goal_wrist_position[2] -= 0.1
            first = True

        error = goal_wrist_position - wrist_position_kinematics
        error = error.reshape(3, 1)
        error = clamp_error(error)

        # print(error)
        # print("================================")
        # print(np.transpose(linear_jacobian) @ error)

        ALPHA = 0.5
        joint_position_increments = ALPHA * np.transpose(linear_jacobian) @ error

        current_joint_positions = robot_handle.get_joint_positions()
        joint_setpoints = {}

        for joint_name, joint_increment in zip(joint_list, joint_position_increments):
            joint_setpoints[joint_name] = (
                current_joint_positions[joint_name] + joint_increment
            )

        robot_handle.set_joint_positions(joint_setpoints)

        print(f"Error = {error}")
        if get_error_mag(error) < 1e-3:
            print("Wrist has successfully reached its position within 1 mm.")
            break

        


if __name__ == "__main__":
    main()
