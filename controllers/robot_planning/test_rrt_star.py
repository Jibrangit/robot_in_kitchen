import numpy as np
import matplotlib.pyplot as plt
from motion_planning import RRT_star, astar_weighted_graph

if __name__ == "__main__":
    np.random.seed(14)

    map = np.load("cspace.npy")
    plt.imshow(map)
    qstart = (50, 70)
    qgoal = (180, 180)

    rrt_star = RRT_star(map, qstart, goal_bias=0.01)
    rrt_star.build_tree(qgoal, plot=True)
    graph = rrt_star.get_tree()
    goal_node = rrt_star.get_goal_node()
    rrt_star.visualize_tree()

    path = []
    if goal_node:
        curr_node = goal_node
        while curr_node != rrt_star._start_node:
            path.append(curr_node)
            curr_node = rrt_star.get_tree()[curr_node].parent
        path.append(rrt_star._start_node)
    curr_node = path[0]
    for path_node in path:
        plt.plot([curr_node[1], path_node[1]], [curr_node[0], path_node[0]], "r-*")
        curr_node = path_node
        plt.pause(0.000001)

    plt.show()
