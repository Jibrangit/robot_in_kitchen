import py_trees
from typing import Union
import time
import numpy as np
import matplotlib.pyplot as plt

from libraries.robot_controller import (
    Controller,
    compute_movt_from_encoder,
    DifferentialDriveRobotParams,
    initialize_robot_params_from_yaml,
)

from libraries.mapping import Mapper, MappingParams

BALL_DIAMETER = 0.0399


class NavigateThroughPoints(py_trees.behaviour.Behaviour):
    """
    Navigate through waypoints using a simple reactive controller that minimizes translational and rotational error.
    If waypoints are not provided, it reads off the current path plan from the blackboard.
    """

    def __init__(
        self,
        waypoints: Union[list, None],
        display: False,
        name: str = "NavigateThroughPoints",
    ):
        super(NavigateThroughPoints, self).__init__(name)
        self.logger.info("%s.__init__()" % (self.__class__.__name__))

        self._waypoints = waypoints

        self._blackboard = self.attach_blackboard_client()
        self._blackboard.register_key(
            key="timestep", access=py_trees.common.Access.READ
        )
        self._blackboard.register_key(
            key="robot_handle", access=py_trees.common.Access.WRITE
        )
        self._blackboard.register_key(key="marker", access=py_trees.common.Access.READ)
        self._blackboard.register_key(
            key="current_plan", access=py_trees.common.Access.READ
        )
        self._blackboard.register_key(
            key="se2_pose", access=py_trees.common.Access.READ
        )

        self._display = display 
        if self._display:
            self._mapper = Mapper(MappingParams("config/mapping_params.yaml"))

    def setup(self, **kwargs: int) -> None:

        self.logger.info("%s.setup()" % (self.__class__.__name__))

        self._timestep = self._blackboard.timestep
        self._robot_handle = self._blackboard.robot_handle
        self._marker = self._blackboard.marker

    def initialise(self) -> None:
        if not self._waypoints:
            self.logger.info(
                "No waypoints provided to behavior upon instantiation, reading the current plan off the blackboard."
            )
            self._waypoints = self._blackboard.current_plan

        robot_params = initialize_robot_params_from_yaml("config/robot_params.yaml")
        self._controller = Controller(robot_params, self._waypoints)

    def update(self) -> py_trees.common.Status:
        """Run the controllers, if controllers have reached all their predefined waypoints, return success."""

        self.logger.debug("%s.update()" % (self.name))

        if self._controller.completed():
            self.logger.info(
                f"Successfully navigated through waypoints for behavior {self.name}"
            )
            return py_trees.common.Status.SUCCESS

        else:
            self._marker.setSFVec3f(
                [
                    *self._controller.get_current_target(),
                    BALL_DIAMETER,
                ]
            )

            # Drive the robot
            xw, yw, theta = self._blackboard.se2_pose

            vl, vr = self._controller.get_input_vels((xw, yw, theta))
            self._robot_handle.set_motors_vels(vl, vr)

            if self._display:
                px, py = self._mapper.world2map(xw, yw)
                plt.plot(py, px, "g*")
                plt.pause(0.00000001)
            return py_trees.common.Status.RUNNING

    def terminate(self, new_status: py_trees.common.Status) -> None:
        self._robot_handle.set_motors_vels(0, 0)
        return super().terminate(new_status)


class MoveLinearly(py_trees.behaviour.Behaviour):
    """
    Move forward by 'x' metres.
    """

    def __init__(
        self,
        wheel_max_speed_radps: float,
        distance: float,
        speed_factor=0.1,
        name: str = "MoveLinearly",
    ):
        super(MoveLinearly, self).__init__(name)
        self.logger.info("%s.__init__()" % (self.__class__.__name__))

        self._wheel_radius = 0.016
        self._wheel_max_speed_radps = wheel_max_speed_radps
        self._distance = distance
        self._travelled_distance = 0.0
        self._speed_factor = speed_factor  # Ratio of max speed

        self._blackboard = self.attach_blackboard_client()
        self._blackboard.register_key(
            key="timestep", access=py_trees.common.Access.READ
        )
        self._blackboard.register_key(
            key="robot_handle", access=py_trees.common.Access.READ
        )

    def setup(self, **kwargs: int) -> None:

        self.logger.info("%s.setup()" % (self.__class__.__name__))

        self._timestep = self._blackboard.timestep
        self._robot_handle = self._blackboard.robot_handle

    def initialise(self) -> None:
        self._prev_el, self._prev_er = self._robot_handle.get_encoder_readings()
        self._dist_moved = 0.0
        self._wheel_speed = (
            (self._distance / np.abs(self._distance))
            * self._wheel_max_speed_radps
            * self._speed_factor
        )

    def update(self) -> py_trees.common.Status:

        el, er = self._robot_handle.get_encoder_readings()
        lin_dist, ang_dist = compute_movt_from_encoder(
            self._prev_el, self._prev_er, el, er, wheel_radius=EFFECTIVE_WHEEL_RADIUS
        )
        self._dist_moved += lin_dist
        self.logger.debug(f"Moved {self._dist_moved} so far")

        if np.abs(self._dist_moved) < np.abs(self._distance):

            self._robot_handle.set_motors_vels(self._wheel_speed, self._wheel_speed)
            return py_trees.common.Status.RUNNING

        else:
            self.logger.info(f"Successfully moved forward by {self._dist_moved} metres")
            self._robot_handle.set_motors_vels(0, 0)
            return py_trees.common.Status.SUCCESS

    def terminate(self, new_status: py_trees.common.Status) -> None:
        return super().terminate(new_status)
