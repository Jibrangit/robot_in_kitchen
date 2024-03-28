import sys
import os
import py_trees

root_dir = os.getenv("HOME") + "/webots/robot_planning"
sys.path.append(root_dir)
sys.path.append(os.getenv('WEBOTS_HOME') + '/lib/controller/python')
from behavior_tree_controller import IsCspaceAvailable

root = py_trees.composites.Sequence(name='multiple_behaviors', memory=True)
failure = py_trees.behaviours.Failure(name='1')
success = py_trees.behaviours.Success(name='1')
root.add_child(success)
root.add_child(IsCspaceAvailable(name='2'))
root.add_child(IsCspaceAvailable(name='3'))
root.add_child(IsCspaceAvailable(name='4'))

while True:
    root.tick_once()