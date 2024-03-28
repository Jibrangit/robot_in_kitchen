"""controller that uses behavior trees"""

import sys
import os

from py_trees.common import Status

root_dir = os.getenv("HOME") + "/webots/robot_planning"
sys.path.append(root_dir)

from copy import deepcopy
import matplotlib.pyplot as plt
import pandas as pd
import py_trees

from controller import Supervisor 
from libraries.robot_device_io import RobotDeviceIO

from mapping_behaviors import MapWithRangeFinder, IsCspaceAvailable
from planning_behaviors import GeneratePath
from navigation_behaviors import NavigateThroughPoints


class BlackboardWriter(py_trees.behaviour.Behaviour):
    """All objects or values needed by other behaviors are written to the blackboard here."""

    def __init__(self, name: str, **kwargs):

        super().__init__(name=name)

        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(
            key="timestep", access=py_trees.common.Access.WRITE
        )
        self.blackboard.register_key(
            key="robot_comms", access=py_trees.common.Access.WRITE
        )
        self.blackboard.register_key(
            key="range_finder", access=py_trees.common.Access.WRITE
        )
        self.blackboard.register_key(
            key="map_display", access=py_trees.common.Access.WRITE
        )
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


def create_tree(**kwargs) -> py_trees.behaviour.Behaviour:

    write_blackboard_variable = BlackboardWriter(name="Writer", **kwargs)
    home_position = (0.4, -3.1)
    clockwise_waypoints = [
        (-1.6, -3.2),
        (-1.65, 0.35),
        (0.65, 0.35),
        (0.67, -1.65),
        (0.56, -3.3),
    ]
    lower_left_position = (-1.6, -3.2)
    sink_position = (0.75, 0.18)

    counter_clockwise_waypoints = deepcopy(clockwise_waypoints)
    counter_clockwise_waypoints.reverse()
    counter_clockwise_waypoints.append(home_position)

    clockwise_navigation = NavigateThroughPoints(
        waypoints=clockwise_waypoints, name="ClockwiseAroundTable"
    )
    counter_clockwise_navigation = NavigateThroughPoints(
        waypoints=counter_clockwise_waypoints, name="CounterClockwiseAroundTable"
    )
    navigation = py_trees.composites.Sequence(
        name="MoveAroundTable",
        memory=True,
        children=[clockwise_navigation, counter_clockwise_navigation],
    )

    generate_map = py_trees.composites.Parallel(
        name="GenerateMap", policy=py_trees.common.ParallelPolicy.SuccessOnOne()
    )

    generate_map.add_children([navigation, MapWithRangeFinder()])

    get_map = py_trees.composites.Selector(
        name="GetMap", memory=True, children=[IsCspaceAvailable(), generate_map]
    )

    compute_path_to_lower_left = GeneratePath(
        goal_position=lower_left_position,
        name="ComputePathToLowerLeft",
    )
    navigate_to_lower_left = NavigateThroughPoints(
        waypoints=None, name="NavigateToLowerLeft"
    )

    compute_path_to_sink = GeneratePath(
        goal_position=sink_position,
        name="ComputePathToSink",
    )
    navigate_to_sink = NavigateThroughPoints(waypoints=None, name="NavigateToSink")

    root = py_trees.composites.Sequence(name="Root behavior", memory=True)
    root.add_children(
        [
            write_blackboard_variable,
            get_map,
            compute_path_to_lower_left,
            navigate_to_lower_left,
            compute_path_to_sink,
            navigate_to_sink,
        ]
    )
    return root


def main():
    robot = Supervisor()
    timestep = int(robot.getBasicTimeStep())

    robot_comms = RobotDeviceIO(robot)
    robot_comms.initialize_devices(timestep)
    robot_comms.set_motors_vels(0, 0)
    range_finder = robot.getDevice("Hokuyo URG-04LX-UG01")

    map_display = robot.getDevice("map_display")
    marker = robot.getFromDef("marker").getField("translation")

    root = create_tree(
        timestep=timestep,
        robot_comms=robot_comms,
        range_finder=range_finder,
        map_display=map_display,
        marker=marker,
    )

    py_trees.display.render_dot_tree(root)
    root.setup_with_descendants()
    root.initialise()

    while robot.step(timestep) != -1:
        if root.status == py_trees.common.Status.SUCCESS:
            break
        root.tick_once()


if __name__ == "__main__":
    main()
