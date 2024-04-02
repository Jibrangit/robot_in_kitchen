import py_trees
from py_trees.common import Status
import numpy as np
import typing as t


class CommandJointPositions(py_trees.behaviour.Behaviour):
    def __init__(self, joint_setpoints: dict, name: str = "CommandJointPositions"):
        super().__init__(name)
        self._blackboard = self.attach_blackboard_client()
        self._joint_setpoints = joint_setpoints

        self._blackboard.register_key(
            key="timestep", access=py_trees.common.Access.READ
        )
        self._blackboard.register_key(
            key="robot_comms", access=py_trees.common.Access.READ
        )

    def setup(self, **kwargs: int) -> None:

        self.logger.info("%s.setup()" % (self.__class__.__name__))

        self._timestep = self._blackboard.timestep
        self._robot_comms = self._blackboard.robot_comms

    def initialise(self) -> None:
        self._robot_comms.set_joint_positions(self._joint_setpoints)

    def update(self) -> Status:
        joint_positions = self._robot_comms.get_joint_positions()
        error = 0
        for position_name in self._joint_setpoints.keys():
            error += abs(
                joint_positions[position_name] - self._joint_setpoints[position_name]
            )

        if error < 0.01:
            self.logger.info(
                f"Successfully commanded robot joints to positions : {joint_positions}"
            )
            return py_trees.common.Status.SUCCESS
        else:
            self.logger.debug(f"Updating joint positions, error = {error}")
            return py_trees.common.Status.RUNNING

    def terminate(self, new_status: Status) -> None:
        return super().terminate(new_status)


class ControlGripper(py_trees.behaviour.Behaviour):
    def __init__(
        self, action: bool, gripping_force: float = None, name: str = "ControlGripper"
    ):
        """
        False is open, True is close.
        """
        super().__init__(name)
        self._blackboard = self.attach_blackboard_client()

        self._blackboard.register_key(
            key="timestep", access=py_trees.common.Access.READ
        )
        self._blackboard.register_key(
            key="robot_comms", access=py_trees.common.Access.READ
        )
        self._action = action
        self._force = gripping_force

    def setup(self, **kwargs: int) -> None:

        self.logger.info("%s.setup()" % (self.__class__.__name__))

        self._timestep = self._blackboard.timestep
        self._robot_comms = self._blackboard.robot_comms

        if self._action:
            self._joint_setpoints = {
                "gripper_left_finger_joint": 0.0,
                "gripper_right_finger_joint": 0.0,
            }
        else:
            self._joint_setpoints = {
                "gripper_left_finger_joint": 0.045,
                "gripper_right_finger_joint": 0.045,
            }

    def initialise(self) -> None:
        self._robot_comms.set_joint_positions(self._joint_setpoints)

    def update(self) -> Status:
        left_force, right_force = self._robot_comms.get_force_feedback()
        if (
            self._action and left_force < self._force and right_force < self._force
        ):  # Force feedback would be negative
            self.logger.info(
                f"Successfully closed gripper with force {-np.sqrt(left_force**2 + right_force**2)} "
            )
            return py_trees.common.Status.SUCCESS
        elif not self._action:
            joint_positions = self._robot_comms.get_joint_positions()
            error = 0
            for position_name in self._joint_setpoints.keys():
                error += abs(
                    joint_positions[position_name]
                    - self._joint_setpoints[position_name]
                )

            if error < 0.01:
                self.logger.info(
                    f"Successfully commanded robot joints to positions : {joint_positions}"
                )
                return py_trees.common.Status.SUCCESS

        else:
            return py_trees.common.Status.RUNNING

    def terminate(self, new_status: Status) -> None:
        return super().terminate(new_status)
