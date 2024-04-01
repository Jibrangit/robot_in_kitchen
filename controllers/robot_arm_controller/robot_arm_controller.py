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
from libraries.behaviors.robot_arm_behaviors import CommandJointPositions
from libraries.behaviors.blackboard_writer import BlackboardWriter
from libraries.behaviors.planning_behaviors import GeneratePath
from libraries.behaviors.navigation_behaviors import NavigateThroughPoints, MoveLinearly

BALL_DIAMETER = 0.0399
WHEEL_MAX_SPEED_RADPS = 10.15


def create_tree(**kwargs) -> py_trees.behaviour.Behaviour:
    write_blackboard_variable = BlackboardWriter(name="Writer", **kwargs)
    home_joint_setpoints = {
        "torso_lift_joint": 0.35,
        "arm_1_joint": 0.71,
        "arm_2_joint": 1.02,
        "arm_3_joint": -2.815,
        "arm_4_joint": 1.011,
        "arm_5_joint": 0,
        "arm_6_joint": 0,
        "arm_7_joint": 0,
        "gripper_left_finger_joint": 0,
        "gripper_right_finger_joint": 0,
        "head_1_joint": 0,
        "head_2_joint": 0,
    }
    pick_joint_positions = None

    with open("robot_config/pre_pick/joint_positions.yaml", "r") as stream:
        pick_joint_positions = yaml.safe_load(stream)

    get_into_pick_position = MoveLinearly(
        wheel_max_speed_radps=WHEEL_MAX_SPEED_RADPS, distance=0.2
    )
    set_pick_joint_positions = CommandJointPositions(pick_joint_positions)

    root = py_trees.composites.Sequence(
        name="CommandRobot",
        memory=True,
        children=[write_blackboard_variable, get_into_pick_position],
    )
    return root


def main():
    robot = Supervisor()
    timestep = int(robot.getBasicTimeStep())

    robot_comms = RobotDeviceIO(robot)
    robot_comms.initialize_devices(timestep)
    robot_comms.set_motors_vels(0, 0)
    range_finder = robot.getDevice("Hokuyo URG-04LX-UG01")

    map_display = robot.getDevice("map_display")
    marker = robot.getFromDef("marker").getField("translation")

    root = create_tree(
        timestep=timestep,
        robot_comms=robot_comms,
        range_finder=range_finder,
        map_display=map_display,
        marker=marker,
    )

    py_trees.display.render_dot_tree(root)
    root.setup_with_descendants()
    root.initialise()

    while robot.step(timestep) != -1:
        if root.status == py_trees.common.Status.SUCCESS:
            break
        root.tick_once()


if __name__ == "__main__":
    main()
