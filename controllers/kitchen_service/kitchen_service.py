"""controller that performs kitchen service"""

import sys
import os
import yaml

from py_trees.common import Status

root_dir = os.getenv("HOME") + "/webots/robot_planning"
sys.path.append(root_dir)

from copy import deepcopy
import matplotlib.pyplot as plt
import pandas as pd
import py_trees

from controller import Supervisor
from libraries.robot_device_io import RobotDeviceIO

from behaviors.blackboard_context import BlackboardContext
from behaviors.publishers import PublishRobotOdometry, PublishRangeFinderData
from behaviors.mapping_behaviors import CheckCspaceExists, MapWithRangeFinder
from libraries.mapping import MappingParams, RangeFinderParams
from behaviors.navigation_behaviors import NavigateThroughPoints
from behaviors.planning_behaviors import GeneratePath


def read_yaml_file(file_path):
    with open(file_path, "r") as file:
        data = yaml.safe_load(file)
        return data


def get_coordinates(data, key):
    return [(entry["x"], entry["y"]) for entry in data[key]]


def create_mapping_tree() -> py_trees.behaviour.Behaviour:

    kitchen_service_positions = read_yaml_file("config/kitchen_service_positions.yaml")
    clockwise_around_table = NavigateThroughPoints(
        get_coordinates(kitchen_service_positions, "clockwise_waypoints"),
        name="ClockwiseAroundTable",
    )
    counter_clockwise_around_table = NavigateThroughPoints(
        get_coordinates(kitchen_service_positions, "counter_clockwise_waypoints"),
        name="CounterClockwiseAroundTable",
    )
    move_around_table = py_trees.composites.Sequence(
        name="MoveAroundTable",
        memory=True,
        children=[clockwise_around_table, counter_clockwise_around_table],
    )
    generate_map = MapWithRangeFinder(display=True)

    map_the_place = py_trees.composites.Parallel(
        name="MapThePlace",
        policy=py_trees.common.ParallelPolicy.SuccessOnOne(),
        children=[move_around_table, generate_map],
    )
    root = py_trees.composites.Selector(
        name="GetConfigurationSpace",
        memory=False,
        children=[CheckCspaceExists(), map_the_place],
    )

    return root


def create_tree(**context) -> py_trees.behaviour.Behaviour:
    blackboard_context = BlackboardContext(name="BlackboardContext", **context)

    publish_odometry = PublishRobotOdometry()
    publish_range_finder_data = PublishRangeFinderData()
    publish_data = py_trees.composites.Parallel(
        name="PublishData",
        policy=py_trees.common.ParallelPolicy.SuccessOnOne(),
        children=[
            publish_odometry,
            publish_range_finder_data,
        ],
    )
    get_configuration_space = create_mapping_tree()

    kitchen_service_positions = read_yaml_file("config/kitchen_service_positions.yaml")
    lower_left_position = get_coordinates(
        kitchen_service_positions, "lower_left_position"
    )[0]
    jar1_robot_position = get_coordinates(
        kitchen_service_positions, "jar1_robot_position"
    )[0]

    plan_to_lower_left = GeneratePath(
        goal_position=lower_left_position, display=True, name="GetPathToLowerLeft"
    )
    navigate_to_lower_left = NavigateThroughPoints(
        waypoints=None, name="NavigateToLowerLeft"
    )
    plan_to_jar1 = GeneratePath(
        goal_position=jar1_robot_position, display=True, name="GetPathToJar1"
    )
    navigate_to_jar1 = NavigateThroughPoints(waypoints=None, name="NavigateToJar1")

    perform_tasks = py_trees.composites.Sequence(name="PerformTasks", memory=True)
    perform_tasks.add_children(
        [
            get_configuration_space,
            plan_to_lower_left,
            navigate_to_lower_left,
            plan_to_jar1,
            navigate_to_jar1,
        ]
    )

    arrange_kitchen = py_trees.composites.Parallel(
        name="ArrangeKitchen",
        policy=py_trees.common.ParallelPolicy.SuccessOnOne(),
        children=[publish_data, perform_tasks],
    )
    kitchen_service = py_trees.composites.Sequence(name="KitchenService", memory=True)
    kitchen_service.add_children(
        [
            blackboard_context,
            arrange_kitchen,
        ]
    )
    return kitchen_service


def main():
    robot = Supervisor()
    timestep = int(robot.getBasicTimeStep())

    # Supply all device handles to communicate with robot and all devices attached to it.
    robot_handle = RobotDeviceIO(robot)
    robot_handle.initialize_devices(timestep)
    robot_handle.set_motors_vels(0, 0)

    range_finder = robot.getDevice("Hokuyo URG-04LX-UG01")
    gps_handle = robot.getDevice("gps")
    compass_handle = robot.getDevice("compass")

    range_finder_params = RangeFinderParams("config/hokuyo_params.yaml")
    mapping_params = MappingParams("config/mapping_params.yaml")

    map_display = robot.getDevice("display")
    marker = robot.getFromDef("marker").getField("translation")

    root = create_tree(
        timestep=timestep,
        robot_handle=robot_handle,
        gps_handle=gps_handle,
        compass_handle=compass_handle,
        range_finder=range_finder,
        range_finder_params=range_finder_params,
        mapping_params=mapping_params,
        display=map_display,
        marker=marker,
    )

    py_trees.display.render_dot_tree(root)
    root.setup_with_descendants()

    while robot.step(timestep) != -1:
        if root.status == py_trees.common.Status.SUCCESS:
            break
        root.tick_once()


if __name__ == "__main__":
    main()
