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

        TORSO_LIFT_INITIAL_POSITION = 0.6
        ARM_1_INITIAL_POSITION = 0.07

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
                    [0, 0, 1, 0], [0, 0, 0]
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
                    joint_initial_position=TORSO_LIFT_INITIAL_POSITION,
                ),
            ),
            Transform(
                "TORSO_LIFT",
                "ARM_FRONT_EXTENSION",
                axis_angle_and_position_to_transformation_matrix(
                    [0, 0, 1, -1.5708], [-0.037, 0.0388, 0.0224]
                ),
            ),
            Transform(
                "ARM_FRONT_EXTENSION",
                "ARM_1_JOINT",
                axis_angle_and_position_to_transformation_matrix(
                    [0, 0, 1, 0.00996], [0.0251, 0.194, -0.171]
                ),
            ),
            # DynamicTransform(parent_frame="ARM_1_JOINT", child_frame="ARM_1", transformation=transformation_matrix_from_rot_axis_and_translation(
            #     rot_angle=0,
            #     rot_axis="y",
            #     trans_vec=[0.125, 0.018, -0.0311],
            # ), joint_name="arm_2_joint",
            # ),
            # Transform(
            #     "ARM_1",
            #     "ARM_2_JOINT",
            #     axis_angle_and_position_to_transformation_matrix(
            #         [1, 0, 0, 0], [0.125, 0.018, -0.0311]
            #     ),
            # ),
            # DynamicTransform(parent_frame="ARM_2_JOINT", child_frame="ARM_2"),
            # Transform(parent_frame="ARM_2", child_frame="ARM_3_JOINT"),
            # DynamicTransform(parent_frame="ARM_3_JOINT", child_frame="ARM_3"),
            # Transform(parent_frame="ARM_3", child_frame="ARM_4_JOINT"),
            # DynamicTransform(parent_frame="ARM_4_JOINT", child_frame="ARM_4"),
            # Transform(parent_frame="ARM_4", child_frame="ARM_5_JOINT"),
            # DynamicTransform(parent_frame="ARM_5_JOINT", child_frame="ARM_5"),
            # Transform(parent_frame="ARM_5", child_frame="ARM_6_JOINT"),
            # DynamicTransform(parent_frame="ARM_6_JOINT", child_frame="ARM_6"),
            # Transform(parent_frame="ARM_6", child_frame="ARM_7_JOINT"),
            # DynamicTransform(parent_frame="ARM_7_JOINT", child_frame="ARM_7"),
            # Transform(parent_frame="ARM_7", child_frame="WRIST"),
            # Transform(
            #     "WRIST",
            #     "WRIST_CONNECTOR",
            #     axis_angle_and_position_to_transformation_matrix(
            #         [-0.57735, -0.57735, -0.57735, 2.0944], [0, 0, 0.012725]
            #     ),
            # ),
            # Transform(
            #     "WRIST_CONNECTOR",
            #     "TIAGO_HAND",
            #     axis_angle_and_position_to_transformation_matrix(
            #         [0, 0, 1, -np.pi / 2], [0, 0, 0]
            #     )
            #     @ axis_angle_and_position_to_transformation_matrix(
            #         [0.57735, 0.57735, -0.57735, 2.0944], [0, 0.016, 0]
            #     ),
            # ),
            # Transform(
            #     parent_frame="TIAGO_HAND", child_frame="GRIPPER_LEFT_FINGER_JOINT"
            # ),
            # Transform(
            #     parent_frame="TIAGO_HAND", child_frame="GRIPPER_RIGHT_FINGER_JOINT"
            # ),
            # DynamicTransform(
            #     parent_frame="GRIPPER_LEFT_FINGER_JOINT",
            #     child_frame="GRIPPER_LEFT_FINGER",
            # ),
            # DynamicTransform(
            #     parent_frame="GRIPPER_RIGHT_FINGER_JOINT",
            #     child_frame="GRIPPER_RIGHT_FINGER",
            # ),
            # Transform(
            #     parent_frame="GRIPPER_LEFT_FINGER",
            #     child_frame="GRIPPER_LEFT_FINGER_ROBOT_END",
            # ),
            # Transform(
            #     parent_frame="GRIPPER_RIGHT_FINGER",
            #     child_frame="GRIPPER_RIGHT_FINGER_ROBOT_END",
            # ),
            # Transform(
            #     parent_frame="GRIPPER_LEFT_FINGER_ROBOT_END",
            #     child_frame="GRIPPER_LEFT_FINGER_HOLDING_END",
            # ),
            # Transform(
            #     parent_frame="GRIPPER_RIGHT_FINGER_ROBOT_END",
            #     child_frame="GRIPPER_RIGHT_FINGER_HOLDING_END",
            # ),
        ]

        self.transform_tree = TransformTree(
            parent_frame="TIAGO_ROBOT", transforms=self._transforms
        )
        self.transform_tree.visualize_transform_tree()
