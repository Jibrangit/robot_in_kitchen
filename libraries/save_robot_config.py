import sys
import os

sys.path.append(os.getenv("WEBOTS_HOME") + "/lib/controller/python")
sys.path.append(os.getenv("HOME") + "/webots/robot_planning")

from controller import Robot, Supervisor
from libraries.robot_device_io import RobotDeviceIO

robot = Supervisor()
timestep = int(robot.getBasicTimeStep())
robot_comms = RobotDeviceIO(robot)
robot_comms.initialize_devices(timestep)

if robot.step(timestep) != -1:
    robot_comms.save_joint_positions("robot_config/pick/joint_positions.yaml")
    robot_comms.save_robot_se2_pose("robot_config/pick/pose.yaml")
