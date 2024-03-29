import py_trees 

class BlackboardWriter(py_trees.behaviour.Behaviour):
    """All objects or values needed by other behaviors are written to the blackboard here."""

    def __init__(self, name: str, **kwargs):

        super().__init__(name=name)

        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(
            key="timestep", access=py_trees.common.Access.WRITE
        )

        # Provides access to the robot's pose (x, y, theta) as well as
        # write-access to set the motor velocities.
        self.blackboard.register_key(
            key="robot_comms", access=py_trees.common.Access.WRITE
        )

        # Provides access to the range-finder device that allows the mapping behaviors to read off the range-finder.
        self.blackboard.register_key(
            key="range_finder", access=py_trees.common.Access.WRITE
        )
        self.blackboard.register_key(
            key="map_display", access=py_trees.common.Access.WRITE
        )

        # Used to access the marker (ping pong ball) and set its position to the latest target for visualization.
        self.blackboard.register_key(key="marker", access=py_trees.common.Access.WRITE)

        self.logger.debug("%s.__init__()" % (self.__class__.__name__))

        self.blackboard.set(name="timestep", value=kwargs["timestep"])
        self.blackboard.set(name="robot_comms", value=kwargs["robot_comms"])
        self.blackboard.set(name="range_finder", value=kwargs["range_finder"])
        self.blackboard.set(name="map_display", value=kwargs["map_display"])
        self.blackboard.set(name="marker", value=kwargs["marker"])

        self.logger.info(
            f'Successfully written all variables to blackboard including the timestep = {kwargs["timestep"]}'
        )

    def setup(self, **kwargs) -> None:
        pass

    def update(self) -> py_trees.common.Status:
        return py_trees.common.Status.SUCCESS
