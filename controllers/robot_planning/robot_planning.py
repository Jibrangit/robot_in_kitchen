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
    PLANNING_AND_EXECUTION = 2


def plan_path(map, start, goal):
    pstart = world2map(start[0], start[1])
    pgoal = world2map(goal[0], goal[1])

    path = astar(map, pstart, pgoal)
    return [map2world(px, py) for px, py in path]


def display_path(map, path):
    pixel_path = [world2map(x, y) for x, y in path]
    plt.imshow(map)
    for p in pixel_path:
        plt.plot(p[1], p[0], "r*")
        plt.show()
        plt.pause(0.000001)


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


class Mapper:
    def __init__(self):
        self._map = np.zeros((MAP_LENGTH, MAP_LENGTH), dtype=float)
        self._pose = (0, 0)

    def world2map(self, x, y) -> Tuple[float]:
        self._pose = (x, y)
        px = np.round(((x - TOP_LEFT_X) / ARENA_WIDTH) * MAP_LENGTH)
        py = np.round(((TOP_LEFT_Y - y) / ARENA_LENGTH) * MAP_LENGTH)

        return int(px), int(py)

    def _is_index_in_bounds(self, px, py):
        return 0 <= px < MAP_LENGTH and 0 <= py < MAP_LENGTH

    def map2world(self, px, py) -> Tuple[float]:
        x = ((px / MAP_LENGTH) * ARENA_WIDTH) + TOP_LEFT_X
        y = TOP_LEFT_Y - ((py / MAP_LENGTH) * ARENA_LENGTH)
        return x, y

    def compute_cspace(self):
        kernel = np.ones((KERNEL_SIZE, KERNEL_SIZE))
        cmap = self._map
        cmap = signal.convolve2d(self._map, kernel, mode="same")
        cmap = np.clip(cmap, 0, 1)  # As convolution increases values to over 1.
        cspace = cmap > OCCUPANCY_GRID_THRESHOLD
        return cspace

    def display_map(self, display):
        # Draw configuration map
        for row in np.arange(0, MAP_LENGTH):
            for col in np.arange(0, MAP_LENGTH):
                v = min(int((self._map[row, col]) * 255), 255)
                if v > 0.01:
                    display.setColor(v * 256**2 + v * 256 + v)
                    display.drawPixel(row, col)

    def display_cspace(self, cspace):
        plt.imshow(cspace)
        plt.show()

    def save_cspace(self, cspace):
        np.save("cspace", cspace)

    def get_map(self):
        return self._map


class RangeFinderMapper(Mapper):
    def __init__(self, lidar):
        super().__init__()
        self._lidar = lidar
        self._laser_line = None

        self._angles = np.linspace(2 * np.pi / 3, -2 * np.pi / 3, LIDAR_NUM_READINGS)
        self._angles = self._angles[LIDAR_FIRST_READING_INDEX:LIDAR_LAST_READING_INDEX]

    def enable_lidar(self, timestep):
        self._lidar.enable(timestep)
        self._lidar.enablePointCloud()

    def _get_lidar_readings(self) -> List[float]:
        ranges = self._lidar.getRangeImage()
        ranges[ranges == np.inf] = 100
        ranges = ranges[LIDAR_FIRST_READING_INDEX:LIDAR_LAST_READING_INDEX]

        return np.array(
            [
                ranges * np.cos(self._angles) + LIDAR_ROBOT_X_OFFSET,
                ranges * np.sin(self._angles),
                np.ones(LIDAR_ACTUAL_NUM_READINGS),
            ]
        )

    def _lidar_robot_to_world(self, xw, yw, theta) -> np.array:
        X_i = self._get_lidar_readings()
        w_T_r = np.array(
            [
                [np.cos(theta), -np.sin(theta), xw],
                [np.sin(theta), np.cos(theta), yw],
                [0, 0, 1],
            ]
        )

        return w_T_r @ X_i

    def generate_map(self, robot_pose) -> None:
        X_w = self._lidar_robot_to_world(robot_pose[0], robot_pose[1], robot_pose[2])
        px_robot, py_robot = self.world2map(robot_pose[0], robot_pose[1])

        for i in range(LIDAR_ACTUAL_NUM_READINGS):
            px, py = self.world2map(X_w[0][i], X_w[1][i])
            if self._map[px, py] < 1:
                self._map[px, py] += 0.01

            # Reduce probability of obstacle for all pixels in the laser's line of sight using Bresenham's algorithm.
            laser_line_coordinates = plot_line(px_robot, py_robot, px, py)
            self._laser_line = laser_line_coordinates
            for coordinate in laser_line_coordinates[1:-1]:
                px_laser = coordinate[0]
                py_laser = coordinate[1]

                if self._map[px_laser, py_laser] > 0.01:
                    self._map[px_laser, py_laser] -= 0.001

    def display_laser_line_of_sight(self, display):
        # Draw configuration map
        for l in self._laser_line:
            display.setColor(0xFF0000)
            display.drawPixel(l[0], l[1])


def main():
    robot = Supervisor()
    timestep = int(robot.getBasicTimeStep())
    robot_comms = RobotDeviceIO(robot)
    robot_comms.initialize_devices(timestep)

    mapper = RangeFinderMapper(robot.getDevice("Hokuyo URG-04LX-UG01"))
    mapper.enable_lidar(timestep)

    map_display = robot.getDevice("map_display")
    cspace_display = robot.getDevice("cspace_display")

    home_position = None
    robot_state = RobotState.IDLE
    mapping_controllers = None
    mapping_controller_idx = 0

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
                mapper.display_cspace(cspace)

                robot_state = RobotState.PLANNING_AND_EXECUTION

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
                # mapper.display_map(map_display)
                # for x, y in zip(xs, ys):
                #     px, py = mapper.world2map(x, y)
                #     map_display.setColor(0x00FF00)
                #     map_display.drawPixel(px, py)

                # Drive the robot
                vl, vr = mapping_controllers[mapping_controller_idx].get_input_vels(
                    (xw, yw, theta)
                )
                robot_comms.set_motors_vels(vl, vr)


if __name__ == "__main__":
    main()
