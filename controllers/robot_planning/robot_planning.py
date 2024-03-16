"""week5_jibran controller."""

from controller import Robot, Supervisor, Motor, PositionSensor
import sys
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


from bresenham import plot_line
from robot_controller import Controller
from motion_planning import astar
from mapping import RangeFinderMapper, MappingParams, RangeFinderParams


LIDAR_NUM_READINGS = 667
LIDAR_ACTUAL_NUM_READINGS = 530
LIDAR_FIRST_READING_INDEX = 57
LIDAR_LAST_READING_INDEX = -80
TOP_LEFT_X = -2.5
TOP_LEFT_Y = 1.8
ARENA_WIDTH = 4.7
ARENA_LENGTH = 5.9
LIDAR_ROBOT_X_OFFSET = 0.202
BALL_DIAMETER = 0.0399
WHEEL_MAX_SPEED_RADPS = 10.15
OCCUPANCY_GRID_THRESHOLD = 0.3
MAP_LENGTH = 300
KERNEL_SIZE = 43


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
        mapping_params=MappingParams(
            MAP_LENGTH,
            ARENA_WIDTH,
            ARENA_LENGTH,
            TOP_LEFT_X,
            TOP_LEFT_Y,
            OCCUPANCY_GRID_THRESHOLD,
            KERNEL_SIZE,
        ),
        range_finder_params=RangeFinderParams(
            LIDAR_NUM_READINGS,
            2 * np.pi / 3,
            -2 * np.pi / 3,
            LIDAR_ACTUAL_NUM_READINGS,
            LIDAR_FIRST_READING_INDEX,
            LIDAR_LAST_READING_INDEX,
            LIDAR_ROBOT_X_OFFSET,
        ),
    )
    mapper.enable_lidar(timestep)

    map_display = robot.getDevice("map_display")
    cspace_display = robot.getDevice("cspace_display")

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
            cspace = np.load("cspace.npy")
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
