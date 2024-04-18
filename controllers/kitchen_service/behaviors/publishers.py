import py_trees
import numpy as np


class PublishRobotOdometry(py_trees.behaviour.Behaviour):

    def __init__(self, name: str = "PublishRobotOdometry"):
        super(PublishRobotOdometry, self).__init__(name)
        self.logger.info("%s.__init__()" % (self.__class__.__name__))

        self._blackboard = self.attach_blackboard_client()
        self._blackboard.register_key(
            key="timestep", access=py_trees.common.Access.READ
        )
        self._blackboard.register_key(
            key="gps_handle", access=py_trees.common.Access.READ
        )
        self._blackboard.register_key(
            key="compass_handle", access=py_trees.common.Access.READ
        )
        self._blackboard.register_key(
            key="se2_pose", access=py_trees.common.Access.WRITE
        )

    def setup(self, **kwargs: int) -> None:

        self.logger.debug("%s.setup()" % (self.__class__.__name__))

        self._timestep = self._blackboard.timestep
        self._gps_handle = self._blackboard.gps_handle
        self._compass_handle = self._blackboard.compass_handle

    def initialise(self) -> None:
        self._gps_handle.enable(self._timestep)
        self._compass_handle.enable(self._timestep)

    def update(self) -> py_trees.common.Status:

        xw = self._gps_handle.getValues()[0]
        yw = self._gps_handle.getValues()[1]
        theta = np.arctan2(self._compass_handle.getValues()[0], self._compass_handle.getValues()[1])

        self._blackboard.set(name="se2_pose", value=(xw, yw, theta))

        print((xw, yw, theta))

        return py_trees.common.Status.RUNNING

    def terminate(self, new_status: py_trees.common.Status) -> None:
        pass

class PublishRangeFinderData(py_trees.behaviour.Behaviour):

    def __init__(self, name: str = "PublishRangeFinderData"):
        super(PublishRangeFinderData, self).__init__(name)
        self.logger.info("%s.__init__()" % (self.__class__.__name__))

        self._blackboard = self.attach_blackboard_client()
        self._blackboard.register_key(
            key="timestep", access=py_trees.common.Access.READ
        )
        self._blackboard.register_key(
            key="range_finder", access=py_trees.common.Access.READ
        )

        self._blackboard.register_key(
            key="range_finder_params", access=py_trees.common.Access.READ
        )   # Range finders specs and how it's mounted on the robot. 
        self._blackboard.register_key(
            key="range_finder_readings", access=py_trees.common.Access.WRITE
        )

    def _enable_lidar(self):
        self._range_finder.enable(self._timestep)
        self._range_finder.enablePointCloud()

    def setup(self, **kwargs: int) -> None:

        self.logger.debug("%s.setup()" % (self.__class__.__name__))

        self._timestep = self._blackboard.timestep
        self._range_finder = self._blackboard.range_finder
        self._range_finder_params = self._blackboard.range_finder_params
        self._angles = np.linspace(
            self._range_finder_params.zero_angle,
            self._range_finder_params.final_angle,
            self._range_finder_params.num_readings,
        )[self._range_finder_params.first_idx : self._range_finder_params.last_idx]

    def initialise(self) -> None:
        self._enable_lidar()

    def update(self) -> py_trees.common.Status:

        ranges = self._range_finder.getRangeImage()
        ranges[ranges == np.inf] = 100
        ranges = ranges[
            self._range_finder_params.first_idx : self._range_finder_params.last_idx
        ]

        range_finder_readings = np.array(
            [
                ranges * np.cos(self._angles) + self._range_finder_params.x_offset,
                ranges * np.sin(self._angles),
                np.ones(self._range_finder_params.actual_num_readings),
            ]
        )
        self._blackboard.set(
            "range_finder_readings", value=range_finder_readings
        )

        print(range_finder_readings)

        return py_trees.common.Status.RUNNING

    def terminate(self, new_status: py_trees.common.Status) -> None:
        pass
