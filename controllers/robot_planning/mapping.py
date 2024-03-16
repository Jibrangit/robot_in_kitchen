import numpy as np
from dataclasses import dataclass
from bresenham import plot_line
import numpy as np
from enum import Enum
from copy import deepcopy
from typing import List, Tuple
from scipy import signal
import matplotlib.pyplot as plt
import math
import pandas as pd
import time


@dataclass
class MappingParams:
    map_length: int
    arena_width: int
    arena_length: int
    top_left_x: float
    top_left_y: float
    occupancy_grid_threshold: float
    kernel_size: float


@dataclass
class RangeFinderParams:
    num_readings: int
    zero_angle: float
    final_angle: float
    actual_num_readings: int
    first_idx: int
    last_idx: int
    x_offset: float


class Mapper:
    def __init__(self, mapping_params: MappingParams):
        self._mapping_params = mapping_params
        self._map = np.zeros(
            (mapping_params.map_length, mapping_params.map_length), dtype=float
        )
        self._pose = (0, 0)

    def world2map(self, x, y) -> Tuple[float]:
        self._pose = (x, y)
        px = np.round(
            ((x - self._mapping_params.top_left_x) / self._mapping_params.arena_width)
            * self._mapping_params.map_length
        )
        py = np.round(
            ((self._mapping_params.top_left_y - y) / self._mapping_params.arena_length)
            * self._mapping_params.map_length
        )

        return int(px), int(py)

    def _is_index_in_bounds(self, px, py):
        return (
            0 <= px < self._mapping_params.map_length
            and 0 <= py < self._mapping_params.map_length
        )

    def map2world(self, px, py) -> Tuple[float]:
        x = (
            (px / self._mapping_params.map_length) * self._mapping_params.arena_width
        ) + self._mapping_params.top_left_x
        y = self._mapping_params.top_left_y - (
            (py / self._mapping_params.map_length) * self._mapping_params.arena_length
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
        for row in np.arange(0, self._mapping_params.map_length):
            for col in np.arange(0, self._mapping_params.map_length):
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
    def __init__(
        self,
        lidar,
        mapping_params: MappingParams,
        range_finder_params: RangeFinderParams,
    ):
        super().__init__(mapping_params)
        self._lidar = lidar
        self._lidar_params = range_finder_params

        self._angles = np.linspace(
            self._lidar_params.zero_angle,
            self._lidar_params.final_angle,
            self._lidar_params.num_readings,
        )
        self._angles = self._angles[
            self._lidar_params.first_idx : self._lidar_params.last_idx
        ]

    def enable_lidar(self, timestep):
        self._lidar.enable(timestep)
        self._lidar.enablePointCloud()

    def _get_lidar_readings(self) -> List[float]:
        ranges = self._lidar.getRangeImage()
        ranges[ranges == np.inf] = 100
        ranges = ranges[self._lidar_params.first_idx : self._lidar_params.last_idx]

        return np.array(
            [
                ranges * np.cos(self._angles) + self._lidar_params.x_offset,
                ranges * np.sin(self._angles),
                np.ones(self._lidar_params.actual_num_readings),
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
