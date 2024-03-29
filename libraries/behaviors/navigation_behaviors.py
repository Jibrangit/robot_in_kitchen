import py_trees 
from typing import Union 

from robot_controller import Controller


WHEEL_MAX_SPEED_RADPS = 10.15
BALL_DIAMETER = 0.0399

class NavigateThroughPoints(py_trees.behaviour.Behaviour):
    """
    Navigate through waypoints using a simple reactive controller that minimizes translational and rotational error.
    """

    def __init__(
        self, waypoints: Union[list, None], name: str = "NavigateThroughPoints"
    ):
        super(NavigateThroughPoints, self).__init__(name)
        self.logger.info("%s.__init__()" % (self.__class__.__name__))

        self._waypoints = waypoints

        self._blackboard_reader = self.attach_blackboard_client()
        self._blackboard_reader.register_key(
            key="timestep", access=py_trees.common.Access.READ
        )
        self._blackboard_reader.register_key(
            key="robot_comms", access=py_trees.common.Access.READ
        )
        self._blackboard_reader.register_key(
            key="marker", access=py_trees.common.Access.READ
        )
        self._blackboard_reader.register_key(
            key="current_plan", access=py_trees.common.Access.READ
        )

    def setup(self, **kwargs: int) -> None:

        self.logger.info("%s.setup()" % (self.__class__.__name__))

        self._timestep = self._blackboard_reader.timestep
        self._robot_comms = self._blackboard_reader.robot_comms
        self._marker = self._blackboard_reader.marker

    def initialise(self) -> None:
        if not self._waypoints:
            self.logger.info(
                "No waypoints provided to behavior upon instantiation, reading the current plan off the blackboard."
            )
            self._waypoints = self._blackboard_reader.current_plan

        self._controller = Controller(WHEEL_MAX_SPEED_RADPS, self._waypoints)

    def update(self) -> py_trees.common.Status:
        """Run the controllers, if controllers have reached all their predefined waypoints, return success."""

        self.logger.debug("%s.update()" % (self.name))

        xw, yw, theta = self._robot_comms.get_se2_pose()

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
            xw, yw, theta = self._robot_comms.get_se2_pose()

            vl, vr = self._controller.get_input_vels((xw, yw, theta))
            self._robot_comms.set_motors_vels(vl, vr)
            return py_trees.common.Status.RUNNING

    def terminate(self, new_status: py_trees.common.Status) -> None:
        self._robot_comms.set_motors_vels(0, 0)
        return super().terminate(new_status)