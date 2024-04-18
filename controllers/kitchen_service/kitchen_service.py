"""controller that performs kitchen service"""

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

from behaviors.blackboard_context import BlackboardContext
from behaviors.publishers import PublishRobotOdometry, PublishRangeFinderData
from libraries.mapping import MappingParams, RangeFinderParams


def create_tree(**context) -> py_trees.behaviour.Behaviour:
    blackboard_context = BlackboardContext(name="BlackboardContext", **context)
    publish_odometry = PublishRobotOdometry()
    publish_range_finder_data = PublishRangeFinderData()

    publish_data = py_trees.composites.Parallel(
        name="PublishData", policy=py_trees.common.ParallelPolicy.SuccessOnOne()
    )
    publish_data.add_children(
        [
            publish_odometry,
            publish_range_finder_data,
        ]
    )
    kitchen_service = py_trees.composites.Sequence(name="KitchenService", memory=True)
    kitchen_service.add_children(
        [
            blackboard_context,
            publish_data,
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
        map_display=map_display,
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
