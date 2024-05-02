import numpy as np
import matplotlib.pyplot as plt
import os
import sys 

root_dir = os.getenv("HOME") + "/webots/robot_planning"
sys.path.append(root_dir)

from libraries.motion_planning import get_weighted_graph_from_map, astar_weighted_graph

if __name__ == "__main__":
    map = np.load("cspace.npy")
    # path = astar(map, (75, 75), (200, 200))
    plt.imshow(map)
    graph = get_weighted_graph_from_map(map, (57, 30))
    path = astar_weighted_graph(graph, (57, 30), (144, 260))

    for path_node in path:
        plt.plot(path_node[1], path_node[0], "r*")
        plt.pause(0.000001)
    plt.show()
