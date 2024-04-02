import numpy as np
from typing import Union
import yaml
import sys
import os
from controller import Robot, Supervisor, Motor, PositionSensor


class RobotDeviceIO:
    def __init__(self, robot: Union[Robot, Supervisor]):
        self._robot = robot
        self._robot_joints = {
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
        }  # Robot joint keys and safe values.

        self._robot_joint_sensors = [
            "torso_lift_joint_sensor",
            "arm_1_joint_sensor",
            "arm_2_joint_sensor",
            "arm_3_joint_sensor",
            "arm_4_joint_sensor",
            "arm_5_joint_sensor",
            "arm_6_joint_sensor",
            "arm_7_joint_sensor",
            "gripper_left_sensor_finger_joint",
            "gripper_right_sensor_finger_joint",
            "head_1_joint_sensor",
            "head_2_joint_sensor",
        ]

        self._sensor_to_joint_name = {
            "torso_lift_joint_sensor": "torso_lift_joint",
            "arm_1_joint_sensor": "arm_1_joint",
            "arm_2_joint_sensor": "arm_2_joint",
            "arm_3_joint_sensor": "arm_3_joint",
            "arm_4_joint_sensor": "arm_4_joint",
            "arm_5_joint_sensor": "arm_5_joint",
            "arm_6_joint_sensor": "arm_6_joint",
            "arm_7_joint_sensor": "arm_7_joint",
            "gripper_left_sensor_finger_joint": "gripper_left_finger_joint",
            "gripper_right_sensor_finger_joint": "gripper_right_finger_joint",
            "head_1_joint_sensor": "head_1_joint",
            "head_2_joint_sensor": "head_2_joint",
        }

    def _initialize_robot_joints(self, timestep):
        self._robot_joint_handles = {}
        for joint in self._robot_joints.keys():
            self._robot_joint_handles[joint] = self._robot.getDevice(joint)

        self._robot_joint_sensor_handles = {}
        for joint_sensor in self._robot_joint_sensors:
            self._robot_joint_sensor_handles[joint_sensor] = self._robot.getDevice(
                joint_sensor
            )
            self._robot_joint_sensor_handles[joint_sensor].enable(timestep)

    def _initialize_force_feedback(self, timestep):
        self._robot_joint_handles["gripper_left_finger_joint"].enableForceFeedback(
            timestep
        )
        self._robot_joint_handles["gripper_right_finger_joint"].enableForceFeedback(
            timestep
        )

    def joints_to_home_positions(self):
        for joint, joint_value in self._robot_joints.items():
            self._robot_joint_handles[joint].setPosition(joint_value)

    def get_joint_positions(self):
        joint_positions = {}
        for joint_sensor, joint_sensor_val in self._robot_joint_sensor_handles.items():
            joint_positions[self._sensor_to_joint_name[joint_sensor]] = (
                self._robot_joint_sensor_handles[joint_sensor].getValue()
            )
        return joint_positions

    def set_joint_positions(self, joint_positions: dict):
        for joint_name, joint_position in joint_positions.items():
            self._robot_joint_handles[joint_name].setPosition(joint_position)

    def get_force_feedback(self):
        return (
            self._robot_joint_handles["gripper_left_finger_joint"].getForceFeedback(),
            self._robot_joint_handles["gripper_right_finger_joint"].getForceFeedback(),
        )

    def initialize_devices(self, timestep) -> None:

        self._leftMotor = self._robot.getDevice("wheel_left_joint")
        self._rightMotor = self._robot.getDevice("wheel_right_joint")
        self._leftMotor.setPosition(float("inf"))
        self._rightMotor.setPosition(float("inf"))

        self._leftMotor.setVelocity(0)
        self._rightMotor.setVelocity(0)

        self._leftEncoder = self._robot.getDevice("wheel_left_joint_sensor")
        self._rightEncoder = self._robot.getDevice("wheel_right_joint_sensor")
        self._leftEncoder.enable(timestep)
        self._rightEncoder.enable(timestep)

        self._gps = self._robot.getDevice("gps")
        self._gps.enable(timestep)

        self._compass = self._robot.getDevice("compass")
        self._compass.enable(timestep)

        self._initialize_robot_joints(timestep)
        self._initialize_force_feedback(timestep)

    def get_se2_pose(self) -> tuple[float]:
        xw = self._gps.getValues()[0]
        yw = self._gps.getValues()[1]
        theta = np.arctan2(self._compass.getValues()[0], self._compass.getValues()[1])

        return xw, yw, theta

    def set_motors_vels(self, vl, vr) -> None:
        self._leftMotor.setVelocity(vl)
        self._rightMotor.setVelocity(vr)

    def save_joint_positions(self, filepath):
        joint_positions = self.get_joint_positions()
        with open(filepath, "w") as outfile:
            yaml.dump(joint_positions, outfile, default_flow_style=False)

    def save_robot_se2_pose(self, filepath):
        pose = list(self.get_se2_pose())
        pose = [float(elem) if isinstance(elem, np.generic) else elem for elem in pose]
        with open(filepath, "w") as outfile:
            yaml.dump(pose, outfile, default_flow_style=False)

    def get_encoder_readings(self) -> tuple[float]:
        return self._leftEncoder.getValue(), self._rightEncoder.getValue()


def load_joint_positions(filepath):
    pass


def load_robot_se2_pose(filepath):
    pass
