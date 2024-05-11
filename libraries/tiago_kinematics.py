import sys
import os

root_dir = os.getenv("HOME") + "/webots/robot_planning"
sys.path.append(root_dir)

import numpy as np

from libraries.transform_tree import *


class TiagoKinematics:
    def __init__(self):
        """
        All frames available wrt any previous frame, typically the robot or a joint.
        """

        TORSO_LIFT_JOINT_POSITION = [0, 0, 0.6]
        ARM_1_INITIAL_POSITION = 0.07

        ARM_1_JOINT_ANCHOR = [0.0251, 0.194, -0.171]
        ARM_2_JOINT_ANCHOR = [0.125, 0.018, -0.031]
        ARM_3_JOINT_ANCHOR = [0.0872, 0, -0.0015]
        ARM_4_JOINT_ANCHOR = [-0.02, -0.027, -0.222]
        ARM_5_JOINT_ANCHOR = [-0.162, 0.02, 0.027]
        ARM_6_JOINT_ANCHOR = [0, 0, 0.15]
        ARM_7_JOINT_ANCHOR = [0.055, 0, 0]

        ARM_1_ENDPOINT_OFFSET = [0, 0, 1, 0.00996365]
        ARM_2_ENDPOINT_OFFSET = [1, 0, 0, np.pi / 2]
        ARM_3_ENDPOINT_OFFSET = [0, 0.7070652786266494, -0.7071482786593564, np.pi]
        ARM_4_ENDPOINT_OFFSET = [
            -0.5773701112195332,
            -0.5773403479610252,
            -0.577340347876871,
            2.0943702371687434,
        ]
        ARM_5_ENDPOINT_OFFSET = [-0, -1, 0, np.pi / 2]
        ARM_6_ENDPOINT_OFFSET = [
            -0.5773496296401605,
            -0.5773505889627373,
            -0.5773505889654484,
            2.094400959325752,
        ]
        ARM_7_ENDPOINT_OFFSET = [
            0.5773505717198315,
            0.5773496641261705,
            0.5773505717223996,
            2.094400907596659,
        ]

        self._transforms = [
            Transform(
                "TIAGO_ROBOT",
                "TORSO",
                axis_angle_and_position_to_transformation_matrix(
                    [0, 0, 1, 0], [-0.054, 0, 0.193]
                ),
            ),
            Transform(
                "TORSO",
                "TORSO_LIFT_JOINT",
                axis_angle_and_position_to_transformation_matrix(
                    [0, 0, 1, 0], TORSO_LIFT_JOINT_POSITION
                ),
            ),
            DynamicTransform(
                parent_frame="TORSO_LIFT_JOINT",
                child_frame="TORSO_LIFT",
                transformation=get_translation_matrix(0, 0, 0),
                joint_params=JointParams(
                    "torso_lift_joint",
                    joint_type="prismatic",
                    joint_axis="z",
                ),
            ),
            Transform(
                "TORSO_LIFT",
                "ARM_FRONT_EXTENSION",
                axis_angle_and_position_to_transformation_matrix(
                    [0, 0, 1, -np.pi / 2], [-0.037, 0.0388, 0.0224]
                ),
            ),
            Transform(
                "ARM_FRONT_EXTENSION",
                "ARM_1_JOINT",
                axis_angle_and_position_to_transformation_matrix(
                    [0, 0, 1, 0], ARM_1_JOINT_ANCHOR
                ),
            ),
            DynamicTransform(
                parent_frame="ARM_1_JOINT",
                child_frame="ARM_1",
                transformation=transformation_matrix_from_rot_axis_and_translation(
                    rot_angle=0,
                    rot_axis="z",
                ),
                joint_params=JointParams(
                    joint_name="arm_1_joint",
                    joint_type="revolute",
                    joint_axis="z",
                    joint_initial_position=ARM_1_INITIAL_POSITION,
                    endpoint_offset=ARM_1_ENDPOINT_OFFSET,
                ),
            ),
            Transform(
                "ARM_1",
                "ARM_2_JOINT",
                axis_angle_and_position_to_transformation_matrix(
                    [1, 0, 0, 0], ARM_2_JOINT_ANCHOR
                ),
            ),
            DynamicTransform(
                parent_frame="ARM_2_JOINT",
                child_frame="ARM_2",
                transformation=transformation_matrix_from_rot_axis_and_translation(
                    rot_angle=0, rot_axis="-y"
                ),
                joint_params=JointParams(
                    joint_name="arm_2_joint",
                    joint_type="revolute",
                    joint_axis="-y",
                    endpoint_offset=ARM_2_ENDPOINT_OFFSET,
                ),
            ),
            Transform(
                "ARM_2",
                "ARM_3_JOINT",
                axis_angle_and_position_to_transformation_matrix(
                    [1, 0, 0, 0], ARM_3_JOINT_ANCHOR
                ),
            ),
            DynamicTransform(
                parent_frame="ARM_3_JOINT",
                child_frame="ARM_3",
                transformation=transformation_matrix_from_rot_axis_and_translation(
                    rot_angle=0, rot_axis="-x"
                ),
                joint_params=JointParams(
                    joint_name="arm_3_joint",
                    joint_type="revolute",
                    joint_axis="-x",
                    endpoint_offset=ARM_3_ENDPOINT_OFFSET,
                ),
            ),
            Transform(
                "ARM_3",
                "ARM_4_JOINT",
                axis_angle_and_position_to_transformation_matrix(
                    [1, 0, 0, 0], ARM_4_JOINT_ANCHOR
                ),
            ),
            DynamicTransform(
                parent_frame="ARM_4_JOINT",
                child_frame="ARM_4",
                transformation=transformation_matrix_from_rot_axis_and_translation(
                    rot_angle=0, rot_axis="y"
                ),
                joint_params=JointParams(
                    joint_name="arm_4_joint",
                    joint_type="revolute",
                    joint_axis="y",
                    endpoint_offset=ARM_4_ENDPOINT_OFFSET,
                ),
            ),
            Transform(
                "ARM_4",
                "ARM_5_JOINT",
                axis_angle_and_position_to_transformation_matrix(
                    [1, 0, 0, 0], ARM_5_JOINT_ANCHOR
                ),
            ),
            DynamicTransform(
                "ARM_5_JOINT",
                "ARM_5",
                transformation_matrix_from_rot_axis_and_translation(
                    rot_angle=0, rot_axis="-x"
                ),
                joint_params=JointParams(
                    joint_name="arm_5_joint",
                    joint_type="revolute",
                    joint_axis="-x",
                    endpoint_offset=ARM_5_ENDPOINT_OFFSET,
                ),
            ),
            Transform(
                "ARM_5",
                "ARM_6_JOINT",
                axis_angle_and_position_to_transformation_matrix(
                    [1, 0, 0, 0], ARM_6_JOINT_ANCHOR
                ),
            ),
            DynamicTransform(
                parent_frame="ARM_6_JOINT",
                child_frame="ARM_6",
                transformation=transformation_matrix_from_rot_axis_and_translation(
                    rot_angle=0, rot_axis="y"
                ),
                joint_params=JointParams(
                    joint_name="arm_6_joint",
                    joint_type="revolute",
                    joint_axis="y",
                    endpoint_offset=ARM_6_ENDPOINT_OFFSET,
                ),
            ),
            Transform(
                "ARM_6",
                "ARM_7_JOINT",
                axis_angle_and_position_to_transformation_matrix(
                    [1, 0, 0, 0], ARM_7_JOINT_ANCHOR
                ),
            ),
            DynamicTransform(
                parent_frame="ARM_7_JOINT",
                child_frame="ARM_7",
                transformation=transformation_matrix_from_rot_axis_and_translation(
                    rot_angle=0, rot_axis="x"
                ),
                joint_params=JointParams(
                    joint_name="arm_7_joint",
                    joint_type="revolute",
                    joint_axis="x",
                    endpoint_offset=ARM_7_ENDPOINT_OFFSET,
                ),
            ),
            Transform("ARM_7", "WRIST", np.eye(4)),
            Transform(
                "WRIST",
                "WRIST_CONNECTOR",
                axis_angle_and_position_to_transformation_matrix(
                    [-0.57735, -0.57735, -0.57735, 2.0944], [0, 0, 0.012725]
                ),
            ),
            Transform(
                "WRIST_CONNECTOR",
                "TIAGO_HAND",
                axis_angle_and_position_to_transformation_matrix(
                    [0, 0, 1, -np.pi / 2], [0, 0, 0]
                )
                @ axis_angle_and_position_to_transformation_matrix(
                    [0.57735, 0.57735, -0.57735, 2.0944], [0, 0.016, 0]
                ),
            ),
            Transform(
                "TIAGO_HAND",
                "LEFT_GRIPPER_JOINT",
                axis_angle_and_position_to_transformation_matrix(
                    [0, 0, 1, np.pi], [0, 0, 0]
                ),
            ),
            Transform(
                "TIAGO_HAND",
                "RIGHT_GRIPPER_JOINT",
                axis_angle_and_position_to_transformation_matrix(
                    [0, 0, -1, 0], [0, 0, 0]
                ),
            ),
            DynamicTransform(
                parent_frame="LEFT_GRIPPER_JOINT",
                child_frame="LEFT_GRIPPER",
                transformation=get_translation_matrix(0, 0, 0),
                joint_params=JointParams(
                    "gripper_left_finger_joint",
                    joint_type="prismatic",
                    joint_axis="-x",
                ),
            ),
            DynamicTransform(
                parent_frame="RIGHT_GRIPPER_JOINT",
                child_frame="RIGHT_GRIPPER",
                transformation=get_translation_matrix(0, 0, 0),
                joint_params=JointParams(
                    joint_name="gripper_right_finger_joint",
                    joint_type="prismatic",
                    joint_axis="x",
                ),
            ),
            Transform(
                "LEFT_GRIPPER",
                "GRIPPER_LEFT_FINGER_MIDPOINT",
                get_translation_matrix(0.004, 0, -0.1741),
            ),
            Transform(
                "RIGHT_GRIPPER",
                "GRIPPER_RIGHT_FINGER_MIDPOINT",
                get_translation_matrix(0.004, 0, -0.1741),
            ),
        ]

        self.transform_tree = TransformTree(
            parent_frame="TIAGO_ROBOT", transforms=self._transforms
        )
        self.transform_tree.visualize_transform_tree()
