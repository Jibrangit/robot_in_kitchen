import numpy as np
from typing import Dict, Tuple, List
import matplotlib.pyplot as plt
import time
from motion_planning import astar_weighted_graph

np.random.seed(14)


class RRT:
    def __init__(self, start_node: Tuple):
        self._start_node = start_node
        self._tree = {self._start_node : []}
        self._iterations = 1000
        self._delta_q = 2.5
        self._last_node = None

    def _add_node_to_tree(self, new_node: Tuple):
        nearest_node = None
        qdist = float("inf")

        for node in self._tree.keys():
            dist = np.sqrt(
                (new_node[0] - node[0]) ** 2 + (new_node[1] - node[1]) ** 2
            )
            if dist < qdist:
                qdist = dist
                nearest_node = node

        self._tree[nearest_node].append((new_node[0], new_node[1], qdist))
        self._tree[new_node] = []

    def build_tree(self, goal):
        start_time = time.time()
        self._goal = goal
        for k in range(self._iterations):
            new_node = (
                np.random.random_sample() * len(map),
                np.random.random_sample() * len(map[0]),
            )
            self._add_node_to_tree(new_node)
            if (
                np.sqrt(
                    (new_node[0] - self._goal[0]) ** 2
                    + (new_node[1] - self._goal[1]) ** 2
                )
                < self._delta_q
            ):
                print(f"Goal Found ==> {new_node} in {time.time() - start_time} seconds!!!!")
                self._last_node = new_node
                break
    
    def get_tree(self):
        return self._tree

    def get_goal_node(self):
        return self._last_node

    def _plot_node(self, node):
        for child in self._tree[node]:
            self._plot_node((child[0], child[1]))
            plt.arrow(node[1], node[0], child[1] - node[1], child[0] - node[0], shape="full", width=0.5)

    def visualize_tree(self):
        plt.plot(self._start_node[1], self._start_node[0], 'r-*', markersize=10)
        plt.plot(self._last_node[1], self._last_node[0], 'g-*', markersize=10)
        self._plot_node(self._start_node)
        
if __name__ == '__main__':
    map = np.ones((200, 300)) * 255
    qstart = (50, 150)
    qgoal = (180, 180)


    rrt = RRT(qstart)
    rrt.build_tree(qgoal)
    graph = rrt.get_tree()
    goal_node = rrt.get_goal_node()
    rrt.visualize_tree()

    if goal_node:
        path = astar_weighted_graph(graph, qstart, goal_node, plot=True)
        curr_node = path[0]
        for path_node in path:
            plt.plot([curr_node[1], path_node[1]], [curr_node[0], path_node[0]], "r-*")
            curr_node = path_node
            plt.pause(0.000001)

    plt.show()
