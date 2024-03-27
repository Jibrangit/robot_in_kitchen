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
import py_trees

from libraries.robot_controller import Controller
from libraries.mapping import RangeFinderMapper, MappingParams, RangeFinderParams
from libraries.robot_device_io import RobotDeviceIO

WHEEL_MAX_SPEED_RADPS = 10.15
BALL_DIAMETER = 0.0399


class GoAroundTableAndMap(py_trees.behaviour.Behaviour):
    """
    Make the robot go around a predefined list of waypoints and map the environment while its moving.
    """

    def __init__(self, name: str = "GoAroundTableAndMap"):
        super(GoAroundTableAndMap, self).__init__(name)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))

        self._home_position = None
        self._clockwise_waypoints = None
        self._counter_clockwise_waypoints = None

        self._blackboard_reader = self.attach_blackboard_client()
        self._blackboard_reader.register_key(
            key="timestep", access=py_trees.common.Access.READ
        )
        self._blackboard_reader.register_key(
            key="robot_comms", access=py_trees.common.Access.READ
        )
        self._blackboard_reader.register_key(
            key="range_finder", access=py_trees.common.Access.READ
        )
        self._blackboard_reader.register_key(
            key="map_display", access=py_trees.common.Access.READ
        )
        self._blackboard_reader.register_key(
            key="marker", access=py_trees.common.Access.READ
        )

    def setup(self, **kwargs: int) -> None:
        """Setup the following :
        - Communication to the robot's sensors and motors.
        - Set up the mapper.
        - Set up the map display.
        """

        self.logger.debug("%s.setup()" % (self.__class__.__name__))

        self._timestep = self._blackboard_reader.timestep
        self._robot_comms = self._blackboard_reader.robot_comms
        self._range_finder = self._blackboard_reader.range_finder
        self._map_display = self._blackboard_reader.map_display
        self._marker = self._blackboard_reader.marker

        self._mapper = RangeFinderMapper(
            self._range_finder,
            mapping_params_filepath="config/mapping_params.yaml",
            range_finder_params_filepath="config/range_finder_params.yaml",
        )
        self._mapper.enable_lidar(self._timestep)

        self._mapping_controllers = None
        self._mapping_controller_idx = 0

    def initialise(self) -> None:
        """Initialize the controllers with the waypoints after getting the robot's start position"""

        xw, yw, theta = self._robot_comms.get_se2_pose()

        if self._home_position == None:
            self._home_position = (xw, yw)
            self._clockwise_waypoints = [
                self._home_position,
                (-1.6, -3.2),
                (-1.65, 0.35),
                (0.65, 0.35),
                (0.67, -1.65),
                (0.56, -3.3),
            ]
            self._counter_clockwise_waypoints = deepcopy(self._clockwise_waypoints)
            self._counter_clockwise_waypoints.reverse()
            self._counter_clockwise_waypoints.append(self._home_position)
            self._mapping_controllers = [
                Controller(WHEEL_MAX_SPEED_RADPS, self._clockwise_waypoints),
                Controller(WHEEL_MAX_SPEED_RADPS, self._counter_clockwise_waypoints),
            ]
            self.logger.debug("Robot is now in mapping mode...")

    def update(self) -> py_trees.common.Status:
        """Run the controllers and map the environment, if all controllers have executed, return success."""

        xw, yw, theta = self._robot_comms.get_se2_pose()

        if self._mapping_controller_idx >= len(self._mapping_controllers):
            self._robot_comms.set_motors_vels(0, 0)

            cspace = self._mapper.compute_cspace()
            self._mapper.save_cspace(cspace)

            return py_trees.common.Status.SUCCESS

        elif self._mapping_controllers[self._mapping_controller_idx].completed():
            self._mapping_controller_idx += 1
            return py_trees.common.Status.RUNNING

        else:
            self._marker.setSFVec3f(
                [
                    *self._mapping_controllers[
                        self._mapping_controller_idx
                    ].get_current_target(),
                    BALL_DIAMETER,
                ]
            )

            # Map the environment
            self._mapper.generate_map((xw, yw, theta))
            self._mapper.display_map(self._map_display)

            # Drive the robot
            vl, vr = self._mapping_controllers[
                self._mapping_controller_idx
            ].get_input_vels((xw, yw, theta))
            self._robot_comms.set_motors_vels(vl, vr)
            return py_trees.common.Status.RUNNING

    def terminate(self, new_status: py_trees.common.Status) -> None:
        """Stop the robot"""

        self._robot_comms.set_motors_vels(0, 0)
        self.logger.debug(
            "%s.terminate()[%s->%s]"
            % (self.__class__.__name__, self.status, new_status)
        )


class IsCspaceAvailable(py_trees.behaviour.Behaviour):
    def __init__(self, name: str = "IsCspaceAvailable"):
        super(IsCspaceAvailable, self).__init__(name)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))

    def setup(self, **kwargs: int) -> None:
        """Nothing to setup here."""
        pass

    def initialise(self) -> None:
        """Nothing to initialise here."""

    def update(self) -> py_trees.common.Status:
        try:
            cspace = np.load("maps/kitchen_cspace.npy")
            return py_trees.common.Status.SUCCESS
        except OSError:
            return py_trees.common.Status.FAILURE

    def terminate(self, new_status: py_trees.common.Status) -> None:
        """Nothing to cleanup here"""
        self.logger.debug(
            "%s.terminate()[%s->%s]"
            % (self.__class__.__name__, self.status, new_status)
        )


class BlackboardWriter(py_trees.behaviour.Behaviour):
    """All objects or values needed by other behaviors are written to the blackboard here."""

    def __init__(self, name: str, **kwargs):

        super().__init__(name=name)

        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(
            key="timestep", access=py_trees.common.Access.WRITE
        )
        self.blackboard.register_key(
            key="robot_comms", access=py_trees.common.Access.WRITE
        )
        self.blackboard.register_key(
            key="range_finder", access=py_trees.common.Access.WRITE
        )
        self.blackboard.register_key(
            key="map_display", access=py_trees.common.Access.WRITE
        )
        self.blackboard.register_key(key="marker", access=py_trees.common.Access.WRITE)

        self.logger.debug("%s.__init__()" % (self.__class__.__name__))

        self.blackboard.set(name="timestep", value=kwargs["timestep"])
        self.blackboard.set(name="robot_comms", value=kwargs["robot_comms"])
        self.blackboard.set(name="range_finder", value=kwargs["range_finder"])
        self.blackboard.set(name="map_display", value=kwargs["map_display"])
        self.blackboard.set(name="marker", value=kwargs["marker"])

        self.logger.info(
            f'Successfully written all variables to blackboarg including the timestep = {kwargs["timestep"]}'
        )

    def setup(self, **kwargs) -> None:
        pass

    def update(self) -> py_trees.common.Status:
        return py_trees.common.Status.SUCCESS


def create_tree(**kwargs) -> py_trees.behaviour.Behaviour:

    write_blackboard_variable = BlackboardWriter(name="Writer", **kwargs)
    get_map = py_trees.composites.Selector(name="Selector", memory=True)
    get_map.add_child(IsCspaceAvailable())
    get_map.add_child(GoAroundTableAndMap())

    root = py_trees.composites.Sequence(name="Root behavior", memory=True)
    root.add_children([write_blackboard_variable, get_map])
    return root


def main():
    robot = Supervisor()
    timestep = int(robot.getBasicTimeStep())

    robot_comms = RobotDeviceIO(robot)
    robot_comms.initialize_devices(timestep)
    robot_comms.set_motors_vels(0, 0)
    range_finder = robot.getDevice("Hokuyo URG-04LX-UG01")

    map_display = robot.getDevice("map_display")
    marker = robot.getFromDef("marker").getField("translation")

    root = create_tree(
        timestep=timestep,
        robot_comms=robot_comms,
        range_finder=range_finder,
        map_display=map_display,
        marker=marker,
    )

    root.setup_with_descendants()
    root.initialise()

    while robot.step(timestep) != -1:
        root.tick_once()


if __name__ == "__main__":
    main()
