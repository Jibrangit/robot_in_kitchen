import typing as t
import numpy as np
from collections import deque, defaultdict
from heapq import heapify, heappush, heappop
import matplotlib.pyplot as plt
from collections import deque
import time


def get_diagonal_neighbors(map: np.array, idx: t.Tuple) -> t.List:
    """
    Returns neighbor indices along with costs to get to them from map[idx]
    """
    width = len(map)
    height = len(map[0])

    neighbor_indexes_costs = [
        (0, 1, 1),
        (0, -1, 1),
        (-1, 0, 1),
        (1, 0, 1),
        (1, 1, np.sqrt(2)),
        (-1, 1, np.sqrt(2)),
        (-1, -1, np.sqrt(2)),
        (1, -1, np.sqrt(2)),
    ]

    neighbors = []
    for n_idx in neighbor_indexes_costs:
        n = (idx[0] + n_idx[0], idx[1] + n_idx[1], n_idx[2])
        if 0 <= n[0] < width and 0 <= n[1] < height:
            if not map[n[0], n[1]]:  # Unoccupied cell
                neighbors.append(n)
    return neighbors


def get_diagonal_distance(x1, y1, x2, y2):
    dx = np.abs(x2 - x1)
    dy = np.abs(y2 - y1)

    if dy < dx:
        return np.sqrt(2) * dy + (dx - dy)
    else:
        return np.sqrt(2) * dx + (dy - dx)


def get_euclidean_distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def get_weighted_graph_from_map(map: np.array, start_node: tuple):
    """
    Parameters
    ----------
    map: Boolean occupancy grid, False is Unoccupied, True is occupied.
    start_node: Node from where to start BFS, for a map to generate a graph to accurately capture C-space, this node must be in that C-space.

    Returns
    -------
    Dict
        Each key is index of the node (x, y) and the value is a list of other nodes along with the cost to get to them (x, y, cost)
    """
    weighted_graph = {}
    visited = set()
    start_time = time.time()

    if map[start_node]:
        print("Start position is non empty!")
        return {}

    bfs_q = deque()
    bfs_q.append(start_node)

    while bfs_q:
        curr_node = bfs_q.popleft()
        if curr_node not in visited:
            neighbors = get_diagonal_neighbors(map, curr_node)
            weighted_graph.update({curr_node: neighbors})
            visited.add(curr_node)
            for neighbor in neighbors:
                bfs_q.append((neighbor[0], neighbor[1]))
    print(f"Weighted graph created in {time.time() - start_time} seconds")
    return weighted_graph


def astar_weighted_graph(
    weighted_graph: dict, start: tuple, goal: tuple, plot=False
) -> list[tuple]:
    start_time = time.time()
    visited = set()  # Set of Tuples
    connections = {}  # Dictionary of tuple (node) and list (parent)

    distances = {}
    for node in weighted_graph:
        distances[(node[0], node[1])] = float("inf")

    distances[start] = 0

    q = []
    heapify(q)

    connections[start] = [list(start)]
    heappush(
        q,
        (
            distances[start]
            + get_euclidean_distance(goal[0], goal[1], start[0], start[1]),
            start,
        ),
    )

    while q:
        curr = heappop(q)  # (Distance from start, node)
        curr_node = curr[1]

        if plot:
            plt.plot(curr_node[1], curr_node[0], 'y-*')
            plt.pause(0.000001)

        if curr_node == goal:
            print(f"Path found in {time.time() - start_time} seconds!")
            path_node = curr_node
            path = []
            while path_node != start:
                path.append(path_node)
                path_node = tuple(connections[path_node])

            path.append(start)
            path.reverse()
            return path

        else:
            neighbors = weighted_graph[
                curr_node
            ]  # List of Tuples[x, y, increment cost]
            for neighbor in neighbors:
                neighbor_idx = (neighbor[0], neighbor[1])
                cost = distances[curr_node] + neighbor[2]

                if (neighbor_idx) not in visited:
                    connections[neighbor_idx] = list(curr_node)
                    distances[neighbor_idx] = cost
                    visited.add(neighbor_idx)
                    heappush(
                        q,
                        (
                            distances[neighbor_idx]
                            + get_euclidean_distance(
                                goal[0], goal[1], neighbor_idx[0], neighbor_idx[1]
                            ),
                            neighbor_idx,
                        ),
                    )

                elif (neighbor_idx) in visited and cost < distances[neighbor_idx]:
                    connections[neighbor_idx] = list(curr_node)
                    distances[neighbor_idx] = cost
                    heappush(
                        q,
                        (
                            distances[neighbor_idx]
                            + get_euclidean_distance(
                                goal[0], goal[1], neighbor_idx[0], neighbor_idx[1]
                            ),
                            neighbor_idx,
                        ),
                    )

                else:
                    continue

    print("Path to goal could not be found!!")
    return []


def astar(map: np.array, start: t.Tuple, goal: t.Tuple) -> t.List[t.Tuple]:
    start_time = time.time()
    visited = set()  # Set of Tuples
    graph = {}  # Dictionary of tuple (node) and list (parent)

    # Use defaultdict for optimization.
    distances = {}
    for i in range(len(map)):
        for j in range(len(map[0])):
            distances[(i, j)] = float("inf")

    distances[start] = 0

    q = []
    heapify(q)

    graph[start] = [list(start)]
    heappush(
        q,
        (
            distances[start]
            + get_euclidean_distance(goal[0], goal[1], start[0], start[1]),
            start,
        ),
    )

    if map[start]:
        print("Start position is non empty!")
        return []

    if map[goal]:
        print("Goal position is non empty!")
        return []

    plt.imshow(map)  # shows the map
    plt.ion()

    while q:
        curr = heappop(q)  # (Distance from start, node)
        curr_node = curr[1]
        if curr_node == goal:
            print(f"Path found in {time.time() - start_time} seconds!")
            path_node = curr_node
            path = []
            while path_node != start:
                path.append(path_node)
                path_node = tuple(graph[path_node])

                # plt.plot(
                #     path_node[1], path_node[0], "r*"
                # )
                # plt.show()
                # plt.pause(0.000001)

            path.reverse()
            return path

        else:
            neighbors = get_diagonal_neighbors(
                map, curr_node
            )  # List of Tuples[x, y, increment cost]
            for neighbor in neighbors:
                neighbor_idx = (neighbor[0], neighbor[1])
                cost = distances[curr_node] + neighbor[2]

                if (neighbor_idx) not in visited:
                    graph[neighbor_idx] = list(curr_node)
                    distances[neighbor_idx] = cost
                    visited.add(neighbor_idx)
                    heappush(
                        q,
                        (
                            distances[neighbor_idx]
                            + get_euclidean_distance(
                                goal[0], goal[1], neighbor_idx[0], neighbor_idx[1]
                            ),
                            neighbor_idx,
                        ),
                    )

                elif (neighbor_idx) in visited and cost < distances[neighbor_idx]:
                    graph[neighbor_idx] = list(curr_node)
                    distances[neighbor_idx] = cost
                    heappush(
                        q,
                        (
                            distances[neighbor_idx]
                            + get_euclidean_distance(
                                goal[0], goal[1], neighbor_idx[0], neighbor_idx[1]
                            ),
                            neighbor_idx,
                        ),
                    )

                else:
                    continue

        # plt.plot(goal[1], goal[0], "y*")  # puts a yellow asterisk at the goal
        # plt.plot(curr_node[1], curr_node[0], "g*")
        # plt.show()
        # plt.pause(0.000001)

    print("Path to goal could not be found!!")
    return []

class RRT:
    def __init__(self, start_node: tuple, goal_bias = 0.1):
        self._start_node = start_node
        self._tree = {self._start_node: []}
        self._iterations = 1000
        self._delta_q = 5
        self._last_node = None
        self._goal_bias = goal_bias
        self._map_length = 300

    def _add_node_to_tree(self, rand_node: tuple, plot=False):
        nearest_node = None
        qdist = float("inf")

        for node in self._tree.keys():
            dist = np.sqrt(
                (rand_node[0] - node[0]) ** 2 + (rand_node[1] - node[1]) ** 2
            )
            if dist < qdist:
                qdist = dist
                nearest_node = node

        nearest_to_new_dist = np.sqrt(
            (rand_node[0] - nearest_node[0]) ** 2
            + (rand_node[1] - nearest_node[1]) ** 2
        )

        step = self._delta_q / nearest_to_new_dist
        new_node = (
            nearest_node[0] + ((rand_node[0] - nearest_node[0]) * step),
            nearest_node[1] + ((rand_node[1] - nearest_node[1]) * step),
        )

        if plot:
            plt.plot(self._start_node[1], self._start_node[0], "r-*", markersize=10)
            plt.plot(self._goal[1], self._goal[0], "g-*", markersize=10)
            plt.arrow(
                nearest_node[1],
                nearest_node[0],
                new_node[1] - nearest_node[1],
                new_node[0] - nearest_node[0],
                shape="full",
                width=0.5,
            )
            plt.pause(0.0000001)

        self._tree[nearest_node].append((new_node[0], new_node[1], qdist))
        self._tree[new_node] = []
        return new_node

    def build_tree(self, goal, plot=False):
        start_time = time.time()
        self._goal = goal

        for k in range(self._iterations):

            rand_node = (
                np.random.random_sample() * self._map_length,
                np.random.random_sample() * self._map_length,
            )
            if np.random.rand() < self._goal_bias:
                rand_node = self._goal

            new_node = self._add_node_to_tree(rand_node, plot)
            if (
                np.sqrt(
                    (new_node[0] - self._goal[0]) ** 2
                    + (new_node[1] - self._goal[1]) ** 2
                )
                < self._delta_q
            ):
                print(
                    f"Goal Found ==> {new_node} in {time.time() - start_time} seconds!!!!"
                )
                self._last_node = new_node
                break

    def get_tree(self):
        return self._tree

    def get_goal_node(self):
        return self._last_node

    def _plot_node(self, node):
        for child in self._tree[node]:
            self._plot_node((child[0], child[1]))
            plt.arrow(
                node[1],
                node[0],
                child[1] - node[1],
                child[0] - node[0],
                shape="full",
                width=0.5,
            )

    def visualize_tree(self):
        plt.plot(self._start_node[1], self._start_node[0], "r-*", markersize=10)
        plt.plot(self._last_node[1], self._last_node[0], "g-*", markersize=10)
        self._plot_node(self._start_node)


