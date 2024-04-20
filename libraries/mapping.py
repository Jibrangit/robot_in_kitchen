import sys
import os

sys.path.append(os.getenv("HOME") + "/webots/robot_planning")

import numpy as np
from dataclasses import dataclass
from libraries.bresenham import plot_line
import numpy as np
from enum import Enum
from copy import deepcopy
from typing import List, Tuple
from scipy import signal
import matplotlib.pyplot as plt
import math
import pandas as pd
import time
import yaml


class MappingParams:
    def __init__(self, filepath) -> None:

        with open(filepath, "r") as mapping_params_file:
            try:
                mapping_params = yaml.safe_load(mapping_params_file)

                self.map_width = mapping_params["map_width"]
                self.map_height = mapping_params["map_height"]
                self.arena_width = mapping_params["arena_width"]
                self.arena_length = mapping_params["arena_length"]
                self.top_left_x = mapping_params["top_left_x"]
                self.top_left_y = mapping_params["top_left_y"]
                self.occupancy_grid_threshold = mapping_params[
                    "occupancy_grid_threshold"
                ]
                self.kernel_size = mapping_params["kernel_size"]

            except yaml.YAMLError as exc:
                print(exc)


class RangeFinderParams:
    def __init__(self, filepath) -> None:

        with open(filepath, "r") as range_finder_params_file:
            try:
                range_finder_params = yaml.safe_load(range_finder_params_file)

                self.num_readings = range_finder_params["num_readings"]
                self.zero_angle = range_finder_params["zero_angle"]
                self.final_angle = range_finder_params["final_angle"]
                self.actual_num_readings = range_finder_params["actual_num_readings"]
                self.first_idx = range_finder_params["first_idx"]
                self.last_idx = range_finder_params["last_idx"]
                self.x_offset = range_finder_params["x_offset"]

            except yaml.YAMLError as exc:
                print(exc)

    def get_data(self) -> dict:
        return {
            "num_readings": self.num_readings,
            "zero_angle": self.zero_angle,
            "final_angle": self.final_angle,
            "actual_num_readings": self.actual_num_readings,
            "first_idx": self.first_idx,
            "last_idx": self.last_idx,
            "x_offset": self.x_offset,
        }


class Mapper:
    def __init__(self, mapping_params_filepath):
        self._mapping_params = MappingParams(mapping_params_filepath)
        self._map = np.zeros(
            (self._mapping_params.map_width, self._mapping_params.map_height),
            dtype=float,
        )
        self._pose = (0, 0)

    def world2map(self, x, y) -> Tuple[float]:
        self._pose = (x, y)
        px = np.round(
            ((x - self._mapping_params.top_left_x) / self._mapping_params.arena_width)
            * self._mapping_params.map_width
        )
        py = np.round(
            ((self._mapping_params.top_left_y - y) / self._mapping_params.arena_length)
            * self._mapping_params.map_height
        )

        return int(px), int(py)

    def _is_index_in_bounds(self, px, py):
        return (
            0 <= px < self._mapping_params.map_width
            and 0 <= py < self._mapping_params.map_height
        )

    def map2world(self, px, py) -> Tuple[float]:
        x = (
            (px / self._mapping_params.map_width) * self._mapping_params.arena_width
        ) + self._mapping_params.top_left_x
        y = self._mapping_params.top_left_y - (
            (py / self._mapping_params.map_height) * self._mapping_params.arena_length
        )
        return x, y

    def compute_cspace(self):
        kernel = np.ones(
            (self._mapping_params.kernel_size, self._mapping_params.kernel_size)
        )
        cmap = self._map
        cmap = signal.convolve2d(self._map, kernel, mode="same")
        cmap = np.clip(cmap, 0, 1)  # As convolution increases values to over 1.
        cspace = cmap > self._mapping_params.occupancy_grid_threshold
        return cspace

    def display_map(self, display):
        # Draw configuration map
        for row in np.arange(0, self._mapping_params.map_width):
            for col in np.arange(0, self._mapping_params.map_height):
                v = min(int((self._map[row, col]) * 255), 255)
                if v > 0.01:
                    display.setColor(v * 256**2 + v * 256 + v)
                    display.drawPixel(row, col)

    def display_cspace(self, cspace):
        plt.imshow(cspace)
        plt.show()

    def save_cspace(self, cspace):
        np.save("maps/kitchen_cspace", cspace)

    def get_map(self):
        return self._map


class RangeFinderMapper(Mapper):
    def __init__(
        self,
        mapping_params: dict,
        range_finder_params: dict,
    ):
        super().__init__(mapping_params)
        self._lidar_params = RangeFinderParams(range_finder_params)

    def _lidar_robot_to_world(self, range_finder_readings, xw, yw, theta) -> np.array:
        X_i = range_finder_readings
        w_T_r = np.array(
            [
                [np.cos(theta), -np.sin(theta), xw],
                [np.sin(theta), np.cos(theta), yw],
                [0, 0, 1],
            ]
        )

        return w_T_r @ X_i

    def generate_map(self, range_finder_readings, robot_pose) -> None:
        X_w = self._lidar_robot_to_world(
            range_finder_readings, robot_pose[0], robot_pose[1], robot_pose[2]
        )
        px_robot, py_robot = self.world2map(robot_pose[0], robot_pose[1])

        for i in range(self._lidar_params.actual_num_readings):
            px, py = self.world2map(X_w[0][i], X_w[1][i])
            if self._map[px, py] < 1:
                self._map[px, py] += 0.01

            # Reduce probability of obstacle for all pixels in the laser's line of sight using Bresenham's algorithm.
            laser_line_coordinates = plot_line(px_robot, py_robot, px, py)
            for coordinate in laser_line_coordinates[1:-1]:
                px_laser = coordinate[0]
                py_laser = coordinate[1]

                if self._map[px_laser, py_laser] > 0.01:
                    self._map[px_laser, py_laser] -= 0.001
