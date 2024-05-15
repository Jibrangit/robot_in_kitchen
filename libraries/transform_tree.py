import sys
import os
import yaml
from dataclasses import dataclass
from typing import Union
import networkx as nx
import matplotlib.pyplot as plt

root_dir = os.getenv("HOME") + "/webots/robot_planning"
sys.path.append(root_dir)

import numpy as np

from libraries.transformations import *


@dataclass
class JointParams:
    joint_name: str
    joint_type: str
    joint_axis: str
    joint_initial_position: float = 0.0
    endpoint_offset: tuple[float] = (0, 0, 1, 0)


@dataclass
class Transform:
    parent_frame: str
    child_frame: str
    transformation: np.ndarray


class DynamicTransform(Transform):
    def __init__(
        self,
        parent_frame: str,
        child_frame: str,
        transformation: np.ndarray,
        joint_params: JointParams,
    ):
        super().__init__(parent_frame, child_frame, transformation)
        self._joint_params = joint_params
        self._initialize_joint_params(joint_params)

    def _initialize_joint_params(self, joint_params: JointParams):
        self._joint_name = joint_params.joint_name
        self._joint_type = joint_params.joint_type
        self._joint_axis = joint_params.joint_axis
        self._joint_initial_position = joint_params.joint_initial_position
        self._endpoint_offset = joint_params.endpoint_offset

    def _get_transform_with_revolute_joint(self, joint_position: float):
        if self._joint_axis == "x":
            transform = np.array(
                [
                    [
                        1,
                        0,
                        0,
                        0,
                    ],
                    [
                        0,
                        np.cos(joint_position),
                        -np.sin(joint_position),
                        0,
                    ],
                    [
                        0,
                        np.sin(joint_position),
                        np.cos(joint_position),
                        0,
                    ],
                    [0, 0, 0, 1],
                ]
            )

        elif self._joint_axis == "-x":
            joint_position = -joint_position
            transform = np.array(
                [
                    [
                        1,
                        0,
                        0,
                        0,
                    ],
                    [
                        0,
                        np.cos(joint_position),
                        -np.sin(joint_position),
                        0,
                    ],
                    [
                        0,
                        np.sin(joint_position),
                        np.cos(joint_position),
                        0,
                    ],
                    [0, 0, 0, 1],
                ]
            )
        elif self._joint_axis == "y":
            transform = np.array(
                [
                    [
                        np.cos(joint_position),
                        0,
                        np.sin(joint_position),
                        0,
                    ],
                    [0, 1, 0, 0],
                    [
                        -np.sin(joint_position),
                        0,
                        np.cos(joint_position),
                        0,
                    ],
                    [0, 0, 0, 1],
                ]
            )

        elif self._joint_axis == "-y":
            joint_position = -joint_position
            transform = np.array(
                [
                    [
                        np.cos(joint_position),
                        0,
                        np.sin(joint_position),
                        0,
                    ],
                    [0, 1, 0, 0],
                    [
                        -np.sin(joint_position),
                        0,
                        np.cos(joint_position),
                        0,
                    ],
                    [0, 0, 0, 1],
                ]
            )
        elif self._joint_axis == "z":
            transform = np.array(
                [
                    [
                        np.cos(joint_position),
                        -np.sin(joint_position),
                        0,
                        0,
                    ],
                    [
                        np.sin(joint_position),
                        np.cos(joint_position),
                        0,
                        0,
                    ],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ]
            )
        elif self._joint_axis == "-z":
            joint_position = -joint_position
            transform = np.array(
                [
                    [
                        np.cos(joint_position),
                        -np.sin(joint_position),
                        0,
                        0,
                    ],
                    [
                        np.sin(joint_position),
                        np.cos(joint_position),
                        0,
                        0,
                    ],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ]
            )
        else:
            raise ValueError(
                "Invalid rotation axis. Use 'x', 'y', 'z', '-x', '-y' or '-z'."
            )

        transform = transform @ axis_angle_and_position_to_transformation_matrix(
            self._endpoint_offset, [0, 0, 0]
        )

        return transform

    def _get_transform_with_prismatic_joint(self, joint_position: float):
        translation_vector = [0, 0, 0]

        if self._joint_axis == "x":
            translation_vector[0] = joint_position
        elif self._joint_axis == "-x":
            translation_vector[0] = -joint_position
        elif self._joint_axis == "y":
            translation_vector[1] = joint_position
        elif self._joint_axis == "-y":
            translation_vector[1] = -joint_position
        elif self._joint_axis == "z":
            translation_vector[2] = joint_position
        elif self._joint_axis == "-z":
            translation_vector[2] = -joint_position

        return np.array(
            [
                [1, 0, 0, translation_vector[0]],
                [0, 1, 0, translation_vector[1]],
                [0, 0, 1, translation_vector[2]],
                [0, 0, 0, 1],
            ]
        )

    def update_transform(self, joint_position: float):
        joint_position -= self._joint_initial_position
        if self._joint_type == "revolute":
            self.transformation = self._get_transform_with_revolute_joint(
                joint_position
            )
        elif self._joint_type == "prismatic":
            self.transformation = self._get_transform_with_prismatic_joint(
                joint_position
            )


class TreeNode:
    def __init__(self, frame_id: str) -> None:
        self._frame_id = frame_id
        self._child_frames = {}

    def get_child_node(self, frame_id):
        for node in self._child_frames.keys():
            if node._frame_id == frame_id:
                return node


class TransformTree:
    def __init__(self, parent_frame: str, transforms: list) -> None:
        self._transforms = transforms
        self._transform_tree = TreeNode(parent_frame)
        print(f"Transform tree initialized with {parent_frame}.")
        self._create_transform_tree()

    def _create_transform_tree(self):
        for transform in self._transforms:
            self._insert_child_frame_to_tree(
                parent_frame_id=transform.parent_frame,
                child_frame_id=transform.child_frame,
                transform=transform,
            )

    def _insert_child_frame_to_tree(
        self,
        parent_frame_id: str,
        child_frame_id: str,
        transform: Transform,
    ):
        """
        - transform : Pose of child frame in parent frame. Ideally, only insert the immediate parent frame.
        """
        if not self._transform_tree._frame_id:
            print(f"Transform tree is empty!")
            return

        queue = [self._transform_tree]
        while len(queue) > 0:
            curr_node = queue.pop(0)
            if curr_node._frame_id == parent_frame_id:
                curr_node._child_frames[TreeNode(child_frame_id)] = transform
                print(f"Frame {child_frame_id} Inserted into tree!!")
                break

            else:
                for child_node in curr_node._child_frames.keys():
                    queue.append(child_node)

    def visualize_transform_tree(self, filename="transform_tree.png"):
        G = nx.DiGraph()
        pos = {}  # Initialize position dictionary

        def generate_centered_integers(size):
            if size % 2 == 0:  # Check if size is even
                half_size = size // 2
                return list(range(-half_size, half_size))
            else:  # If size is odd, center at zero
                half_size = size // 2
                return list(range(-half_size, half_size + 1))

        def add_nodes_recursively(node, parent_frame_id=None, pos_x=0, pos_y=0):
            if parent_frame_id:
                G.add_edge(parent_frame_id, node._frame_id)
            if node._frame_id not in pos:
                pos[node._frame_id] = (
                    pos_x,
                    pos_y,
                )  # Assign position based on level and y-coordinate

            pos_y += 1
            num_child_frames = len(node._child_frames)
            indices = generate_centered_integers(num_child_frames)

            for idx, (child_frame, transform) in enumerate(node._child_frames.items()):
                add_nodes_recursively(
                    child_frame, node._frame_id, pos_x + indices[idx], pos_y
                )

        add_nodes_recursively(self._transform_tree)

        plt.figure(figsize=(10, 8))

        # Invert y-coordinates for better visualization
        for node_id, (x, y) in pos.items():
            pos[node_id] = (x, -y)

        nx.draw(
            G, pos, with_labels=True, node_size=3000, node_color="skyblue", font_size=10
        )
        plt.savefig(filename)
        plt.show()

    def _find_node(self, node: TreeNode, frame: str):
        if node._frame_id == frame:
            return node

        elif node._child_frames.keys():
            for node_child in node._child_frames.keys():
                node = self._find_node(node_child, frame)
                if node:
                    return node

        else:
            return

    def _get_joint_properties(self, joint_frame_id: str) -> JointParams:
        joint_node = self._find_node(self._transform_tree, joint_frame_id)
        joint_controlled_node = joint_node.get_child_node(joint_frame_id[:-6])
        joint_to_joint_controlled_transform = joint_node._child_frames[
            joint_controlled_node
        ]

        return joint_to_joint_controlled_transform._joint_params

    def update_joint_controlled_transform(self, frame_id: str, joint_position: float):
        joint_node = self._find_node(self._transform_tree, frame_id + "_JOINT")
        joint_controlled_node = joint_node.get_child_node(frame_id)
        joint_node._child_frames[joint_controlled_node].update_transform(joint_position)

    def get_pose(self, parent_frame: str, child_frame: str) -> np.ndarray:
        def get_transform_wrt_parent_node(
            parent_node: TreeNode, child_frame: str
        ) -> np.ndarray:
            if parent_node._frame_id == child_frame:
                return np.eye(4)

            elif parent_node._child_frames:
                for child_node, transform in parent_node._child_frames.items():
                    transform_wrt_curr_child = get_transform_wrt_parent_node(
                        child_node, child_frame
                    )
                    if transform_wrt_curr_child is not None:
                        return transform.transformation @ transform_wrt_curr_child

            else:
                return None

        parent_node = self._find_node(self._transform_tree, parent_frame)
        return get_transform_wrt_parent_node(parent_node, child_frame)

    def get_joint_axis_vector_wrt_robot(self, joint_frame_id: str):

        rot_matrix = self.get_pose(self._transform_tree._frame_id, joint_frame_id)[
            :3, :3
        ]

        joint_params = self._get_joint_properties(joint_frame_id)
        joint_axis = joint_params.joint_axis
        axis_vector = np.array([0, 0, 0])

        if joint_axis == "x":
            axis_vector[0] = 1
        elif joint_axis == "-x":
            axis_vector[0] = -1
        elif joint_axis == "y":
            axis_vector[1] = 1
        elif joint_axis == "-y":
            axis_vector[1] = -1
        elif joint_axis == "z":
            axis_vector[2] = 1
        elif joint_axis == "-z":
            axis_vector[2] = -1

        return rot_matrix @ axis_vector

    def get_joint_jacobian(
        self, joint_frame_id: str, endpoint_frame_id: str
    ) -> np.array:
        joint_params = self._get_joint_properties(joint_frame_id)
        joint_axis_vector = self.get_joint_axis_vector_wrt_robot(joint_frame_id)

        if joint_params.joint_type == "prismatic":
            return np.concatenate((joint_axis_vector, [0, 0, 0]))

        elif joint_params.joint_type == "revolute":
            joint_axis_position = self.get_pose(
                self._transform_tree._frame_id, joint_frame_id
            )[:3, -1]
            endpoint_position = self.get_pose(
                self._transform_tree._frame_id, endpoint_frame_id
            )[:3, -1]

            return np.concatenate(
                (
                    np.cross(
                        joint_axis_vector, endpoint_position - joint_axis_position
                    ),
                    joint_axis_vector,
                )
            )

    def get_complete_jacobian(self, endpoint_frame_id: str) -> tuple[list, np.array]:
        def get_jacobian_from_node(
            curr_node: TreeNode,
            curr_jacobian: np.ndarray,
            endpoint_frame_id: str,
            joint_list: list[str],
        ):
            if curr_node._frame_id == endpoint_frame_id:
                return curr_jacobian, joint_list

            if "_JOINT" in curr_node._frame_id:
                curr_jacobian = np.vstack(
                    (
                        curr_jacobian,
                        self.get_joint_jacobian(curr_node._frame_id, endpoint_frame_id),
                    )
                )

                joint_params = self._get_joint_properties(curr_node._frame_id)
                joint_list.append(joint_params.joint_name)

            for child_node in curr_node._child_frames.keys():
                jacobian = get_jacobian_from_node(
                    child_node, curr_jacobian, endpoint_frame_id, joint_list
                )

                if jacobian is not None:
                    return jacobian

            return

        jacobian, joint_list = get_jacobian_from_node(
            self._transform_tree, np.empty((0, 6)), endpoint_frame_id, []
        )

        return np.transpose(jacobian), joint_list
