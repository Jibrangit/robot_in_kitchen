"""controller that uses behavior trees"""

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


from libraries.bresenham import plot_line
from libraries.robot_controller import Controller
from libraries.motion_planning import astar
from libraries.mapping import RangeFinderMapper, MappingParams, RangeFinderParams
from libraries.robot_device_io import RobotDeviceIO

def main():
    robot = Supervisor()
    timestep = int(robot.getBasicTimeStep())
    robot_comms = RobotDeviceIO(robot)
    robot_comms.initialize_devices(timestep)

if __name__ == "__main__":
    main()
