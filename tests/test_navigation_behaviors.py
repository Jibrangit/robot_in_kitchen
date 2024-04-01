import sys
import os

root_dir = os.getenv("HOME") + "/webots/robot_planning"
sys.path.append(root_dir)


from libraries.robot_controller import LinearMovementController 
from controller import Robot 


