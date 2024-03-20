import numpy as np
import matplotlib.pyplot as plt
from motion_planning import RRT, astar_weighted_graph

if __name__ == "__main__":
    np.random.seed(14)

    map = np.load("cspace.npy")
    plt.imshow(map)
    qstart = (50, 70)
    qgoal = (180, 180)

    rrt = RRT(map, qstart, goal_bias=0.01)
    rrt.build_tree(qgoal, plot=False)
    graph = rrt.get_tree()
    goal_node = rrt.get_goal_node()
    rrt.visualize_tree()

    if goal_node:
        path = astar_weighted_graph(graph, qstart, goal_node, plot=False)
        curr_node = path[0]
        for path_node in path:
            plt.plot([curr_node[1], path_node[1]], [curr_node[0], path_node[0]], "r-*")
            curr_node = path_node
            plt.pause(0.000001)

    plt.show()
