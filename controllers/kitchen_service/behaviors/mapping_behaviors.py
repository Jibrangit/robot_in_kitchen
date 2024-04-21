import py_trees
import numpy as np
from py_trees.common import Status
from libraries.mapping import RangeFinderMapper, MappingParams, RangeFinderParams


class CheckCspaceExists(py_trees.behaviour.Behaviour):
    def __init__(self, name: str = "CheckCspaceExists"):
        super(CheckCspaceExists, self).__init__(name)
        self.logger.info("%s.__init__()" % (self.__class__.__name__))

    def initialise(self) -> None:
        self.logger.debug("%s.initialise()" % (self.__class__.__name__))
        return super().initialise()

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


class MapWithRangeFinder(py_trees.behaviour.Behaviour):
    """
    Make the robot go around a predefined list of waypoints and map the environment while its moving.
    """

    def __init__(
        self,
        display: bool,
        name: str = "MapWithRangeFinder",
    ):
        super(MapWithRangeFinder, self).__init__(name)
        self.logger.info("%s.__init__()" % (self.__class__.__name__))
        self._blackboard = self.attach_blackboard_client()
        self._blackboard.register_key(
            key="timestep", access=py_trees.common.Access.READ
        )
        self._blackboard.register_key(
            key="range_finder_params", access=py_trees.common.Access.READ
        )
        self._blackboard.register_key(
            key="mapping_params", access=py_trees.common.Access.READ
        )
        self._blackboard.register_key(
            key="range_finder_readings", access=py_trees.common.Access.READ
        )
        self._blackboard.register_key(
            key="se2_pose", access=py_trees.common.Access.READ
        )
        self._blackboard.register_key(
            key="display", access=py_trees.common.Access.WRITE
        )

        self._display = display

    def setup(self, **kwargs: int) -> None:

        self.logger.debug("%s.setup()" % (self.__class__.__name__))

        self._timestep = self._blackboard.timestep

        self._mapper = RangeFinderMapper(
            mapping_params=MappingParams("config/mapping_params.yaml"),
            range_finder_params=RangeFinderParams("config/hokuyo_params.yaml"),
        )

    def update(self) -> py_trees.common.Status:

        xw, yw, theta = self._blackboard.se2_pose
        self._mapper.generate_map(
            range_finder_readings=self._blackboard.range_finder_readings,
            robot_pose=(xw, yw, theta),
        )
        self.logger.debug(f"{self.name}.update(), Map generating....")

        if self._display:
            self._mapper.display_map(self._blackboard.display)

        return py_trees.common.Status.RUNNING


    def terminate(self, new_status: py_trees.common.Status) -> None:
        """
        Save the cspace from the map generated so far.
        """
        cspace = self._mapper.compute_cspace()
        self._mapper.save_cspace(cspace)
        if self._display:
            self._mapper.display_cspace(cspace)

        self.logger.info(
            "%s.terminate()[%s->%s], Cspace saved."
            % (self.__class__.__name__, self.status, new_status)
        )
