import py_trees
import yaml





class BlackboardContext(py_trees.behaviour.Behaviour):
    """All objects or values needed by other behaviors are written to the _blackboard here."""

    def __init__(self, name: str, **kwargs):

        super().__init__(name=name)

        self._blackboard = self.attach_blackboard_client()
        self._blackboard.register_key(
            key="timestep", access=py_trees.common.Access.WRITE
        )
        self._blackboard.register_key(
            key="robot_handle", access=py_trees.common.Access.WRITE
        )
        self._blackboard.register_key(
            key="gps_handle", access=py_trees.common.Access.WRITE
        )
        self._blackboard.register_key(
            key="compass_handle", access=py_trees.common.Access.WRITE
        )
        self._blackboard.register_key(
            key="range_finder", access=py_trees.common.Access.WRITE
        )
        self._blackboard.register_key(
            key="range_finder_params", access=py_trees.common.Access.WRITE
        )
        self._blackboard.register_key(
            key="mapping_params", access=py_trees.common.Access.WRITE
        )
        self._blackboard.register_key(
            key="display", access=py_trees.common.Access.WRITE
        )
        self._blackboard.register_key(
            key="marker", access=py_trees.common.Access.WRITE
        )  # Used to access the marker (ping pong ball) and set its position to the latest target for visualization.


        self.logger.debug("%s.__init__()" % (self.__class__.__name__))

        self._blackboard.set(name="timestep", value=kwargs["timestep"])
        self._blackboard.set(name="robot_handle", value=kwargs["robot_handle"])
        self._blackboard.set(name="gps_handle", value=kwargs["gps_handle"])
        self._blackboard.set(name="compass_handle", value=kwargs["compass_handle"])
        self._blackboard.set(name="range_finder", value=kwargs["range_finder"])
        self._blackboard.set(name="mapping_params", value=kwargs["mapping_params"])
        self._blackboard.set(
            name="range_finder_params", value=kwargs["range_finder_params"]
        )
        self._blackboard.set(name="display", value=kwargs["display"])
        self._blackboard.set(name="marker", value=kwargs["marker"])

        self.logger.info(
            f'Written context to _blackboard including the timestep = {kwargs["timestep"]}, and range finder params = {kwargs["range_finder_params"].get_data()}'
        )

    def update(self) -> py_trees.common.Status:
        return py_trees.common.Status.SUCCESS
