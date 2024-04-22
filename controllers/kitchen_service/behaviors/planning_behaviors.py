import py_trees
import numpy as np
import matplotlib.pyplot as plt
from libraries.mapping import Mapper, MappingParams
from libraries.motion_planning import astar
import math


def angle_between_points(p1: tuple, p2: tuple, p3: tuple):
    """Calculate the angle between three points."""
    angle1 = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
    angle2 = math.atan2(p3[1] - p2[1], p3[0] - p2[0])
    angle = abs(angle1 - angle2)
    return angle if angle <= math.pi else 2 * math.pi - angle


def reduce_points(points: list[tuple], threshold_angle: float):
    """Reduce a list of points by keeping only those where the angle between line segments exceeds the threshold."""
    reduced_points = [points[0]]  # Keep the first point
    for i in range(1, len(points) - 1):
        angle = angle_between_points(points[i - 1], points[i], points[i + 1])
        if angle >= threshold_angle:
            reduced_points.append(points[i])
    reduced_points.append(points[-1])  # Keep the last point
    return reduced_points


class GeneratePath(py_trees.behaviour.Behaviour):
    def __init__(
        self,
        goal_position: tuple,
        display: bool,
        name: str = "GeneratePath",
    ):
        super(GeneratePath, self).__init__(name)
        self.logger.info("%s.__init__()" % (self.__class__.__name__))

        self._goal_position = goal_position
        self._mapper = Mapper(MappingParams("config/mapping_params.yaml"))

        self._blackboard = self.attach_blackboard_client()
        self._blackboard.register_key(
            key="current_plan", access=py_trees.common.Access.WRITE
        )
        self._blackboard.register_key(
            key="se2_pose", access=py_trees.common.Access.READ
        )

        self._plan = None
        self._display = display

    def _display_plan(self, p_start, p_goal):
        if self._plan:
            plt.clf()
            plt.imshow(self._cspace)
            plt.plot(p_start[1], p_start[0], "y*")
            plt.pause(0.00000001)
            plt.plot(p_goal[1], p_goal[0], "g*")
            plt.pause(0.00000001)
            x_values = [point[1] for point in self._plan]
            y_values = [point[0] for point in self._plan]
            plt.plot(x_values, y_values, "b-")
            plt.pause(0.00000001)

    def initialise(self) -> None:
        self.logger.info(f"Planning path for behavior {self.name}")

        self.logger.info("%s.initialise()" % (self.name))
        try:
            self._cspace = np.load("maps/kitchen_cspace.npy")

        except OSError:
            self.logger.error("No Cspace available for path planning!")

        xw, yw, theta = self._blackboard.se2_pose

        self._home_position = (xw, yw)

        p_start = self._mapper.world2map(self._home_position[0], self._home_position[1])
        p_goal = self._mapper.world2map(self._goal_position[0], self._goal_position[1])

        self._plan = astar(self._cspace, p_start, p_goal)
        self._plan = reduce_points(points=self._plan, threshold_angle=0.1)

        if self._display:
            self._display_plan(p_start, p_goal)

        for idx, pose in enumerate(self._plan):
            self._plan[idx] = self._mapper.map2world(pose[0], pose[1])

    def update(self) -> py_trees.common.Status:

        if self._plan:
            self._blackboard.set(name="current_plan", value=self._plan, overwrite=True)

            return py_trees.common.Status.SUCCESS
        else:
            return py_trees.common.Status.FAILURE
