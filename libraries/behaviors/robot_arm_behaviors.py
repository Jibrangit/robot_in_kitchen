import py_trees
from py_trees.common import Status
import numpy as np


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

    def update(self) -> Status:
        self._robot_comms.set_joint_positions(self._joint_setpoints)
        joint_positions = self._robot_comms.get_joint_positions()
        error = 0
        for position, setpoint in zip(
            joint_positions.values(), self._joint_setpoints.values()
        ):
            error += (position - setpoint) ** 2

        if error < 0.01:
            self.logger.info(
                f"Successfully commanded robot joints to positions : {self._joint_setpoints}"
            )
            return py_trees.common.Status.SUCCESS
        else:
            self.logger.info(f"Updating joint positions, error = {error}")
            return py_trees.common.Status.RUNNING

    def terminate(self, new_status: Status) -> None:
        return super().terminate(new_status)
