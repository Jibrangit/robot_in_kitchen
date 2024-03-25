import numpy as np 
from typing import Union
from controller import Robot, Supervisor, Motor, PositionSensor


class RobotDeviceIO:
    def __init__(self, robot : Union[Robot, Supervisor]):
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

    def get_se2_pose(self) -> tuple[float]:
        xw = self._gps.getValues()[0]
        yw = self._gps.getValues()[1]
        theta = np.arctan2(self._compass.getValues()[0], self._compass.getValues()[1])

        return xw, yw, theta

    def set_motors_vels(self, vl, vr) -> None:
        self._leftMotor.setVelocity(vl)
        self._rightMotor.setVelocity(vr)