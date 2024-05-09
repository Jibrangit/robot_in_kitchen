import sys
import os

import numpy as np

root_dir = os.getenv("HOME") + "/webots/robot_planning"
sys.path.append(root_dir)

from libraries.transform_tree import *


def main():
    transforms = [
        Transform(
            "frame_0",
            "frame_1_JOINT",
            np.eye(4),
        ),
        DynamicTransform(
            "frame_1_JOINT",
            "frame_1",
            np.eye(4),
            JointParams("joint_1", "prismatic", "z", joint_initial_position=0.1),
        ),
        Transform(
            "frame_0",
            "frame_2",
            np.array([[1, 0, 0, 2], [0, 1, 0, 2], [0, 0, 1, 2], [0, 0, 0, 1]]),
        ),
        Transform(
            "frame_0",
            "frame_3",
            np.array([[1, 0, 0, 3], [0, 1, 0, 3], [0, 0, 1, 3], [0, 0, 0, 1]]),
        ),
        Transform(
            "frame_2",
            "frame_4",
            np.array([[1, 0, 0, 4], [0, 1, 0, 4], [0, 0, 1, 4], [0, 0, 0, 1]]),
        ),
        DynamicTransform(
            "frame_3",
            "frame_5",
            np.array([[1, 0, 0, 5], [0, 1, 0, 5], [0, 0, 1, 5], [0, 0, 0, 1]]),
            JointParams(joint_name="joint_3", joint_type="prismatic", joint_axis="z"),
        ),
        Transform(
            "frame_5",
            "frame_6",
            np.array([[1, 0, 0, 6], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
        ),
        Transform(
            "frame_6",
            "frame_7_JOINT",
            np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 3], [0, 0, 0, 1]]),
        ),
        DynamicTransform(
            "frame_7_JOINT",
            "frame_7",
            np.eye(4),
            JointParams(
                joint_name="joint_7",
                joint_type="revolute",
                joint_axis="x",
                anchor=[0, 2, 0],
            ),
        ),
    ]

    tf_tree = TransformTree("frame_0", transforms)
    # tf_tree.visualize_transform_tree()

    assert (
        np.linalg.det(
            tf_tree.get_pose(parent_frame="frame_0", child_frame="frame_5")
            - np.array([[1, 0, 0, 15], [0, 1, 0, 15], [0, 0, 1, 15], [0, 0, 0, 1]])
        )
        == 0
    )

    assert (
        np.linalg.det(
            tf_tree.get_pose(parent_frame="frame_0", child_frame="frame_1") - np.eye(4)
        )
        == 0
    )
    tf_tree.update_joint_controlled_transform("frame_1", 0.5)
    prismatic_frame = np.eye(4)
    prismatic_frame[2, -1] = 0.4
    assert (
        np.linalg.det(
            tf_tree.get_pose(parent_frame="frame_0", child_frame="frame_1")
            - prismatic_frame
        )
        == 0
    )

    tf_tree.update_joint_controlled_transform(
        frame_id="frame_7", joint_position=np.pi / 4
    )
    assert np.linalg.det(
        tf_tree.get_pose(parent_frame="frame_5", child_frame="frame_7")
        - np.array(
            [
                [1, 0, 0, 6],
                [0, np.cos(np.pi / 4), -np.sin(np.pi / 4), 2],
                [0, np.sin(np.pi / 4), np.cos(np.pi / 4), 3],
                [0, 0, 0, 1],
            ]
        )
    ) == 0


if __name__ == "__main__":
    main()
