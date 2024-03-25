"""week5_jibran controller."""

import sys
import os

root_dir = os.getenv("HOME") + "/webots/robot_planning"
sys.path.append(root_dir)


from controller import Robot, Supervisor, Motor, PositionSensor
from pathlib import Path
import numpy as np
from enum import Enum
from copy import deepcopy
from typing import List, Tuple
from scipy import signal
import matplotlib.pyplot as plt
import math
import pandas as pd
import time


from libraries.bresenham import plot_line
from libraries.robot_controller import Controller
from libraries.motion_planning import astar
from libraries.mapping import RangeFinderMapper, MappingParams, RangeFinderParams

BALL_DIAMETER = 0.0399
WHEEL_MAX_SPEED_RADPS = 10.15


class RobotState(Enum):
    IDLE = 0
    MAPPING = 1
    PLANNING = 2
    NAVIGATION = 3


class RobotDeviceIO:
    def __init__(self, robot):
        self._robot = robot

    def initialize_devices(self, timestep) -> None:

        self._leftMotor = self._robot.getDevice("wheel_left_joint")
        self._rightMotor = self._robot.getDevice("wheel_right_joint")
        self._leftMotor.setPosition(float("inf"))
        self._rightMotor.setPosition(float("inf"))

        # leftEncoder = robot.getDevice('wheel_left_joint_sensor')
        # rightEncoder = robot.getDevice('wheel_right_joint_sensor')
        # leftEncoder.enable(timestep)
        # rightEncoder.enable(timestep)

        self._gps = self._robot.getDevice("gps")
        self._gps.enable(timestep)

        self._compass = self._robot.getDevice("compass")
        self._compass.enable(timestep)

    def get_se2_pose(self) -> Tuple[float]:
        xw = self._gps.getValues()[0]
        yw = self._gps.getValues()[1]
        theta = np.arctan2(self._compass.getValues()[0], self._compass.getValues()[1])

        return xw, yw, theta

    def set_motors_vels(self, vl, vr) -> None:
        self._leftMotor.setVelocity(vl)
        self._rightMotor.setVelocity(vr)


def main():
    robot = Supervisor()
    timestep = int(robot.getBasicTimeStep())
    robot_comms = RobotDeviceIO(robot)
    robot_comms.initialize_devices(timestep)

    mapper = RangeFinderMapper(
        robot.getDevice("Hokuyo URG-04LX-UG01"),
        mapping_params_filepath="config/mapping_params.yaml",
        range_finder_params_filepath="config/range_finder_params.yaml",
    )
    mapper.enable_lidar(timestep)

    map_display = robot.getDevice("map_display")

    home_position = None
    goal_position = (-1.65, 0.0)
    robot_state = RobotState.IDLE
    planned = False

    mapping_controllers = None
    mapping_controller_idx = 0

    planning_controller = None

    marker = robot.getFromDef("marker").getField("translation")
    xs = []
    ys = []

    while robot.step(timestep) != -1:
        xw, yw, theta = robot_comms.get_se2_pose()
        xs.append(xw)
        ys.append(yw)

        if home_position == None:
            home_position = (xw, yw)
            WP = [
                home_position,
                (-1.6, -3.2),
                (-1.65, 0.35),
                (0.65, 0.35),
                (0.67, -1.65),
                (0.56, -3.3),
            ]
            reverse_WP = deepcopy(WP)
            reverse_WP.reverse()
            reverse_WP.append(home_position)
            robot_state = RobotState.MAPPING
            mapping_controllers = [
                Controller(WHEEL_MAX_SPEED_RADPS, WP),
                Controller(WHEEL_MAX_SPEED_RADPS, reverse_WP),
            ]
            mapping_controller_idx = 0
            print("Robot is now in mapping mode...")

        if robot_state == RobotState.MAPPING and mapping_controllers:
            if mapping_controller_idx >= len(mapping_controllers):
                robot_comms.set_motors_vels(0, 0)

                cspace = mapper.compute_cspace()
                mapper.save_cspace(cspace)

                robot_state = RobotState.PLANNING

            elif mapping_controllers[mapping_controller_idx].completed():
                mapping_controller_idx += 1

            else:
                marker.setSFVec3f(
                    [
                        *mapping_controllers[
                            mapping_controller_idx
                        ].get_current_target(),
                        BALL_DIAMETER,
                    ]
                )

                # Map the environment
                mapper.generate_map((xw, yw, theta))
                mapper.display_map(map_display)

                # Drive the robot
                vl, vr = mapping_controllers[mapping_controller_idx].get_input_vels(
                    (xw, yw, theta)
                )
                robot_comms.set_motors_vels(vl, vr)

        if robot_state == RobotState.PLANNING:
            cspace = np.load("maps/kitchen_cspace.npy")
            p_start = mapper.world2map(home_position[0], home_position[1])
            p_goal = mapper.world2map(goal_position[0], goal_position[1])

            plan = astar(cspace, p_start, p_goal)
            for idx, pose in enumerate(plan):
                plan[idx] = mapper.map2world(pose[0], pose[1])

            planning_controller = Controller(WHEEL_MAX_SPEED_RADPS, plan)
            robot_state = RobotState.NAVIGATION

        if robot_state == RobotState.NAVIGATION and planning_controller:
            if planning_controller.completed():
                robot_comms.set_motors_vels(0.0, 0.0)
                break

            else:
                vl, vr = planning_controller.get_input_vels((xw, yw, theta))
                robot_comms.set_motors_vels(vl, vr)


if __name__ == "__main__":
    main()
