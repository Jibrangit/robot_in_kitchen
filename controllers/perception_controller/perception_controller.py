"""robot_arm_controller controller."""

import sys
import os

root_dir = os.getenv("HOME") + "/webots/robot_planning"
sys.path.append(root_dir)

from controller import Robot, Supervisor, Motor, PositionSensor
from pathlib import Path
import numpy as np
from enum import Enum
from copy import deepcopy
from typing import List, Tuple, Union
from scipy import signal
import matplotlib.pyplot as plt
import math
import pandas as pd
import time
import py_trees
import yaml


from libraries.robot_controller import Controller
from libraries.robot_device_io import RobotDeviceIO
from libraries.behaviors.robot_arm_behaviors import (
    CommandJointPositions,
    ControlGripper,
)
from libraries.behaviors.blackboard_writer import BlackboardWriter
from libraries.behaviors.planning_behaviors import GeneratePath
from libraries.behaviors.navigation_behaviors import NavigateThroughPoints, MoveLinearly
from libraries.transformations import (
    get_transformation_matrix,
    get_transformation_matrix_with_zero_rotation,
    axis_angle_and_position_to_transformation_matrix,
)

BALL_DIAMETER = 0.0399
WHEEL_MAX_SPEED_RADPS = 10.15


def create_tree(**kwargs) -> py_trees.behaviour.Behaviour:
    write_blackboard_variable = BlackboardWriter(name="Writer", **kwargs)
    pick_joint_positions = None
    pickup_joint_positions = None

    JC = {
        "openGripper": {
            "gripper_left_finger_joint": 0.045,
            "gripper_right_finger_joint": 0.045,
        },
        "closeGripper": {
            "gripper_left_finger_joint": 0.0,
            "gripper_right_finger_joint": 0.0,
        },
    }

    with open("robot_config/pick/joint_positions.yaml", "r") as stream:
        pick_joint_positions = yaml.safe_load(stream)
    with open("robot_config/pickup/joint_positions.yaml", "r") as stream:
        pickup_joint_positions = yaml.safe_load(stream)

    get_into_pick_position = MoveLinearly(
        wheel_max_speed_radps=WHEEL_MAX_SPEED_RADPS, distance=0.15
    )
    set_pick_joint_positions = CommandJointPositions(pick_joint_positions)
    close_gripper = ControlGripper(action=True, gripping_force=-10)
    pickup_jar = CommandJointPositions(pickup_joint_positions)

    root = py_trees.composites.Sequence(
        name="CommandRobot",
        memory=True,
        children=[
            write_blackboard_variable,
            set_pick_joint_positions,
            get_into_pick_position,
            close_gripper,
            pickup_jar,
        ],
    )
    return root


def main():
    robot = Supervisor()
    timestep = int(robot.getBasicTimeStep())

    robot_comms = RobotDeviceIO(robot)
    robot_comms.initialize_devices(timestep)
    robot_comms.set_motors_vels(0, 0)
    range_finder = robot.getDevice("Hokuyo URG-04LX-UG01")

    display = robot.getDevice("display")

    camera = robot.getDevice("camera")

    camera.enable(timestep)
    camera.recognitionEnable(timestep)
    display.attachCamera(camera)

    robot_to_torso = get_transformation_matrix(theta=0, d=0.193, a=-0.054, alpha=0)

    torso_to_torso_lift_joint = get_transformation_matrix(
        theta=0, d=-0.0005, a=0, alpha=0
    )

    while robot.step(timestep) != 1:
        objects = camera.getRecognitionObjects()
        joint_positions = robot_comms.get_joint_positions()
        d0 = joint_positions["torso_lift_joint"]
        theta1 = joint_positions["head_1_joint"]
        theta2 = joint_positions["head_2_joint"]

        torso_lift_joint_to_torso_lift = get_transformation_matrix(
            theta=0, d=d0 + 0.6, a=0, alpha=0
        )
        torso_lift_to_neck_joint = get_transformation_matrix(
            theta=0, d=0, a=0.182, alpha=0
        )
        neck_joint_to_neck = get_transformation_matrix(
            theta=theta1, d=0.0, a=0.0, alpha=0.0
        )
        neck_to_head_joint = get_transformation_matrix(
            theta=0, d=0.098, a=0.000254, alpha=1.58784  # COnfused here.
        )
        head_joint_to_head = get_transformation_matrix(theta=theta2, d=0, a=0, alpha=0)
        head_to_camera_pose = get_transformation_matrix_with_zero_rotation(
            x=0.107, y=0.0802, z=0
        )
        camera_pose_to_camera = get_transformation_matrix(
            theta=0, d=0, a=0, alpha=-np.pi / 2
        )

        robot_to_camera = (
            robot_to_torso
            @ torso_to_torso_lift_joint
            @ torso_lift_joint_to_torso_lift
            @ torso_lift_to_neck_joint
            @ neck_joint_to_neck
            @ neck_to_head_joint
            @ head_joint_to_head
            @ head_to_camera_pose
            @ camera_pose_to_camera
        )

        object_positions_in_robot_frame = []

        for object in objects:
            orientation = list(object.getOrientation())
            position = list(object.getPosition())

            camera_to_object = axis_angle_and_position_to_transformation_matrix(
                orientation, position
            )

            object_positions_in_robot_frame.append(robot_to_camera @ camera_to_object)
            print(object_positions_in_robot_frame[0])
            print("===============================================")


if __name__ == "__main__":
    main()
