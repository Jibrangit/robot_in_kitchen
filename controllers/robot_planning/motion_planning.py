import typing as t
import numpy as np
from collections import deque, defaultdict
from heapq import heapify, heappush, heappop
import matplotlib.pyplot as plt
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
            if not map[n[0], n[1]]:
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


if __name__ == "__main__":
    map = np.load("cspace.npy")
    path = astar(map, (75, 75), (200, 200))
