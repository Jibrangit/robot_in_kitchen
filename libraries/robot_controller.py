import numpy as np
from enum import Enum
from typing import List, Tuple
from dataclasses import dataclass
import yaml 

TRACKWIDTH = 0.054
WHEEL_RADIUS = 0.0201


class RobotDriveState(Enum):
    IDLE = 0
    DRIVE = 1
    TURN = 2
    ADJUST_CASTOR = 3


@dataclass
class DifferentialDriveRobotParams:
    wheel_max_speed_radps: float
    wheel_radius: float
    effective_wheel_radius: float
    trackwidth: float

def initialize_robot_params_from_yaml(robot_params_file_path : str):
    with open(robot_params_file_path, 'r') as file:
        params = yaml.safe_load(file)
    
    return DifferentialDriveRobotParams(**params)

class Controller:
    def __init__(self, robot_params: DifferentialDriveRobotParams, waypoints) -> None:
        self._waypoints = waypoints
        self._index = 0
        self._robot_state = RobotDriveState.DRIVE
        self._MIN_POSITION_ERROR = 0.3  # metres
        self._MIN_HEADING_ERROR = 0.1  # radians
        self._robot_params = robot_params

    def compute_errors(self, pose) -> Tuple[float]:
        """
        pose : SE2 ; x, y, theta in world coordinates
        """
        xw = pose[0]
        yw = pose[1]
        theta = pose[2]

        rho = np.sqrt(
            (xw - self._waypoints[self._index][0]) ** 2
            + (yw - self._waypoints[self._index][1]) ** 2
        )
        alpha = (
            np.arctan2(
                self._waypoints[self._index][1] - yw,
                self._waypoints[self._index][0] - xw,
            )
            - theta
        )

        # atan2 discontinuity
        if alpha > np.pi:
            alpha -= 2 * np.pi

        if alpha < -np.pi:
            alpha += 2 * np.pi

        return rho, alpha

    def get_input_vels(self, pose) -> Tuple[float]:
        """
        pose : SE2 ; x, y, theta in world coordinates
        """

        rho, alpha = self.compute_errors(pose)

        if self._robot_state == RobotDriveState.DRIVE:

            p_trans, p_rot = (
                0.3 * self._robot_params.wheel_max_speed_radps,
                0.1 * self._robot_params.wheel_max_speed_radps,
            )
            vl = p_trans * rho - p_rot * alpha
            vr = p_trans * rho + p_rot * alpha

            if abs(rho) < self._MIN_POSITION_ERROR:
                self._index += 1
                self._robot_state = RobotDriveState.TURN

        elif self._robot_state == RobotDriveState.TURN:

            p_trans, p_rot = (
                0.1 * self._robot_params.wheel_max_speed_radps,
                0.3 * self._robot_params.wheel_max_speed_radps,
            )
            vl = p_trans * rho - p_rot * alpha
            vr = p_trans * rho + p_rot * alpha

            if abs(alpha) < self._MIN_HEADING_ERROR:
                self._robot_state = RobotDriveState.DRIVE

        else:
            vl, vr = 0.0, 0.0

        vl = max(
            min(vl, self._robot_params.wheel_max_speed_radps),
            -self._robot_params.wheel_max_speed_radps,
        )
        vr = max(
            min(vr, self._robot_params.wheel_max_speed_radps),
            -self._robot_params.wheel_max_speed_radps,
        )

        return vl, vr

    def completed(self) -> bool:
        """
        Controller has traversed all the waypoints provided to it.
        """
        if self._index >= len(self._waypoints):
            return True
        else:
            return False

    def get_index(self) -> int:
        return self._index

    def get_current_target(self) -> float:
        return self._waypoints[self._index]


def compute_movt_from_encoder(
    prev_el, prev_er, el, er, robot_params: DifferentialDriveRobotParams
) -> Tuple[float, float]:
    delta_l = (el - prev_el) * robot_params.effective_wheel_radius
    delta_r = (er - prev_er) * robot_params.effective_wheel_radius

    delta_s = (delta_l + delta_r) / 2.0
    delta_omega_z = (delta_r - delta_l) / robot_params.trackwidth

    return delta_s, delta_omega_z
