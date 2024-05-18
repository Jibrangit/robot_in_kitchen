import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

class BoundingBox:
    def __init__(self, leftSouthLower : tuple, rightNorthUpper : tuple):
        self.leftSouthLower = leftSouthLower
        self.rightNorthUpper = rightNorthUpper

    def contains(self, point: tuple):
        return all(self.leftSouthLower[i] <= point[i] <= self.rightNorthUpper[i] for i in range(3))

    def intersects(self, other):
        return all(self.leftSouthLower[i] <= other.rightNorthUpper[i] and self.rightNorthUpper[i] >= other.leftSouthLower[i] for i in range(3))

class Octree:
    def __init__(self, leftSouthLower: tuple, rightNorthUpper: tuple, capacity=8):
        self.leftSouthLower = leftSouthLower
        self.rightNorthUpper = rightNorthUpper
        self._bounding_box = BoundingBox(leftSouthLower, rightNorthUpper)
        self._capacity = capacity
        self._points = []
        self._subdivided = False

    def insert(self, point):
        if not self._bounding_box.contains(point):
            return False

        if len(self._points) < self._capacity:
            self._points.append(point)
            return True
        else:
            if not self._subdivided:
                self._subdivide()

            return (
                self.leftSouthLowerTree.insert(point) or
                self.rightSouthLowerTree.insert(point) or
                self.leftNorthLowerTree.insert(point) or
                self.rightNorthLowerTree.insert(point) or
                self.leftSouthUpperTree.insert(point) or
                self.rightSouthUpperTree.insert(point) or
                self.leftNorthUpperTree.insert(point) or
                self.rightNorthUpperTree.insert(point)
            )

    def _subdivide(self):
        mid_x = (self.leftSouthLower[0] + self.rightNorthUpper[0]) / 2
        mid_y = (self.leftSouthLower[1] + self.rightNorthUpper[1]) / 2
        mid_z = (self.leftSouthLower[2] + self.rightNorthUpper[2]) / 2

        self.leftSouthLowerTree = Octree(
            self.leftSouthLower,
            (mid_x, mid_y, mid_z),
            self._capacity
        )
        
        self.rightSouthLowerTree = Octree(
            (mid_x, self.leftSouthLower[1], self.leftSouthLower[2]),
            (self.rightNorthUpper[0], mid_y, mid_z),
            self._capacity
        )
        
        self.leftNorthLowerTree = Octree(
            (self.leftSouthLower[0], mid_y, self.leftSouthLower[2]),
            (mid_x, self.rightNorthUpper[1], mid_z),
            self._capacity
        )
        
        self.rightNorthLowerTree = Octree(
            (mid_x, mid_y, self.leftSouthLower[2]),
            (self.rightNorthUpper[0], self.rightNorthUpper[1], mid_z),
            self._capacity
        )
        
        self.leftSouthUpperTree = Octree(
            (self.leftSouthLower[0], self.leftSouthLower[1], mid_z),
            (mid_x, mid_y, self.rightNorthUpper[2]),
            self._capacity
        )
        
        self.rightSouthUpperTree = Octree(
            (mid_x, self.leftSouthLower[1], mid_z),
            (self.rightNorthUpper[0], mid_y, self.rightNorthUpper[2]),
            self._capacity
        )
        
        self.leftNorthUpperTree = Octree(
            (self.leftSouthLower[0], mid_y, mid_z),
            (mid_x, self.rightNorthUpper[1], self.rightNorthUpper[2]),
            self._capacity
        )
        
        self.rightNorthUpperTree = Octree(
            (mid_x, mid_y, mid_z),
            self.rightNorthUpper,
            self._capacity
        )

        self._subdivided = True

    def _get_all_bounding_boxes(self):
        boxes = [self._bounding_box]
        if self._subdivided:
            boxes.extend(self.leftSouthLowerTree._get_all_bounding_boxes())
            boxes.extend(self.rightSouthLowerTree._get_all_bounding_boxes())
            boxes.extend(self.leftNorthLowerTree._get_all_bounding_boxes())
            boxes.extend(self.rightNorthLowerTree._get_all_bounding_boxes())
            boxes.extend(self.leftSouthUpperTree._get_all_bounding_boxes())
            boxes.extend(self.rightSouthUpperTree._get_all_bounding_boxes())
            boxes.extend(self.leftNorthUpperTree._get_all_bounding_boxes())
            boxes.extend(self.rightNorthUpperTree._get_all_bounding_boxes())
        return boxes

def visualize_octree(octree):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    def draw_bounding_box(ax, bbox):
        lsl = bbox.leftSouthLower
        rnu = bbox.rightNorthUpper

        points = np.array([
            [lsl[0], lsl[1], lsl[2]], [rnu[0], lsl[1], lsl[2]], [rnu[0], rnu[1], lsl[2]], [lsl[0], rnu[1], lsl[2]],
            [lsl[0], lsl[1], rnu[2]], [rnu[0], lsl[1], rnu[2]], [rnu[0], rnu[1], rnu[2]], [lsl[0], rnu[1], rnu[2]]
        ])
        edges = [
            [points[0], points[1]], [points[1], points[2]], [points[2], points[3]], [points[3], points[0]],
            [points[4], points[5]], [points[5], points[6]], [points[6], points[7]], [points[7], points[4]],
            [points[0], points[4]], [points[1], points[5]], [points[2], points[6]], [points[3], points[7]]
        ]
        for edge in edges:
            ax.plot3D(*zip(*edge), color="b")

    boxes = octree._get_all_bounding_boxes()
    for box in boxes:
        draw_bounding_box(ax, box)

    points = [point for subtree in [octree.leftSouthLowerTree, octree.rightSouthLowerTree, octree.leftNorthLowerTree,
                                    octree.rightNorthLowerTree, octree.leftSouthUpperTree, octree.rightSouthUpperTree,
                                    octree.leftNorthUpperTree, octree.rightNorthUpperTree]
              if subtree is not None
              for point in subtree._points]
    if octree._points:
        points.extend(octree._points)
        
    points = np.array(points)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='r')

    plt.show()

def main():
    octree = Octree((0, 0, 0), (10, 10, 10), capacity=1)
    points = [(1, 1, 1), (9, 9, 9), (5, 5, 5), (6, 6, 6), (7, 7, 7), (3, 3, 3), (4, 4, 4), (2, 2, 2)]
    for point in points:
        octree.insert(point)

    visualize_octree(octree)

if __name__ == "__main__":
    main()