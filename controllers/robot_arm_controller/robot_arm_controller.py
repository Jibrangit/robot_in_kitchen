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
from libraries.behaviors.robot_arm_behaviors import CommandJointPositions, ControlGripper
from libraries.behaviors.blackboard_writer import BlackboardWriter
from libraries.behaviors.planning_behaviors import GeneratePath
from libraries.behaviors.navigation_behaviors import NavigateThroughPoints, MoveLinearly

BALL_DIAMETER = 0.0399
WHEEL_MAX_SPEED_RADPS = 10.15


def create_tree(**kwargs) -> py_trees.behaviour.Behaviour:
    write_blackboard_variable = BlackboardWriter(name="Writer", **kwargs)
    pick_joint_positions = None
    pickup_joint_positions = None 

    JC = { 'openGripper' : {'gripper_left_finger_joint' : 0.045,
                        'gripper_right_finger_joint': 0.045},
       'closeGripper': {'gripper_left_finger_joint' : 0.0,
                        'gripper_right_finger_joint': 0.0}}

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
            pickup_jar
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
