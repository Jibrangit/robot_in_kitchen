import numpy as np 
import py_trees
from libraries.mapping import RangeFinderMapper


class MapWithRangeFinder(py_trees.behaviour.Behaviour):
    """
    Make the robot go around a predefined list of waypoints and map the environment while its moving.
    """

    def __init__(self, name: str = "MapWithRangeFinder"):
        super(MapWithRangeFinder, self).__init__(name)
        self.logger.info("%s.__init__()" % (self.__class__.__name__))

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
        """
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

    def update(self) -> py_trees.common.Status:

        xw, yw, theta = self._robot_comms.get_se2_pose()
        self._mapper.generate_map((xw, yw, theta))
        self._mapper.display_map(self._map_display)
        self.logger.debug(f"{self.name}.update(), Map generating....")
        return py_trees.common.Status.RUNNING

    def terminate(self, new_status: py_trees.common.Status) -> None:
        """
        Save the cspace from the map generated so far.
        """
        cspace = self._mapper.compute_cspace()
        self._mapper.save_cspace(cspace)
        self.logger.info(
            "%s.terminate()[%s->%s], Cspace saved."
            % (self.__class__.__name__, self.status, new_status)
        )


class IsCspaceAvailable(py_trees.behaviour.Behaviour):
    def __init__(self, name: str = "IsCspaceAvailable"):
        super(IsCspaceAvailable, self).__init__(name)
        self.logger.info("%s.__init__()" % (self.__class__.__name__))

    def initialise(self) -> None:
        self.logger.info("%s.initialise()" % (self.__class__.__name__))
        return super().initialise()

    def update(self) -> py_trees.common.Status:
        try:
            cspace = np.load("maps/kitchen_cspace.npy")
            return py_trees.common.Status.SUCCESS
        except OSError:
            return py_trees.common.Status.FAILURE

    def terminate(self, new_status: py_trees.common.Status) -> None:
        """Nothing to cleanup here"""
        self.logger.info(
            "%s.terminate()[%s->%s]"
            % (self.__class__.__name__, self.status, new_status)
        )
