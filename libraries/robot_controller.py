import numpy as np
from enum import Enum
from typing import List, Tuple
class RobotDriveState(Enum):
    IDLE = 0
    DRIVE = 1
    TURN = 2
    ADJUST_CASTOR = 3

class Controller:
    def __init__(self, wheel_max_speed : float, waypoints) -> None:
        self._waypoints = waypoints
        self._index = 0
        self._robot_state = RobotDriveState.DRIVE
        self._MIN_POSITION_ERROR = 0.3  # metres
        self._MIN_HEADING_ERROR = 0.1   # radians
        self._wheel_max_speed_radps = wheel_max_speed

    def compute_errors(self, pose) -> Tuple[float]:
        '''
            pose : SE2 ; x, y, theta in world coordinates
        '''
        xw = pose[0]
        yw = pose[1]
        theta = pose[2]

        rho = np.sqrt((xw - self._waypoints[self._index][0])**2 + (yw - self._waypoints[self._index][1])**2)
        alpha = np.arctan2(self._waypoints[self._index][1] - yw, self._waypoints[self._index][0] - xw) - theta
        
        # atan2 discontinuity
        if alpha > np.pi:
            alpha -= 2 * np.pi
        
        if alpha < -np.pi:
            alpha += 2 * np.pi

        return rho, alpha

    def get_input_vels(self, pose) -> Tuple[float]:
        '''
            pose : SE2 ; x, y, theta in world coordinates
        '''

        rho, alpha = self.compute_errors(pose)

        if self._robot_state == RobotDriveState.DRIVE:

            p_trans, p_rot = 0.3 * self._wheel_max_speed_radps, 0.1 * self._wheel_max_speed_radps
            vl = p_trans * rho - p_rot * alpha 
            vr = p_trans * rho + p_rot * alpha

            if abs(rho) < self._MIN_POSITION_ERROR:
                self._index += 1
                self._robot_state = RobotDriveState.TURN


        elif self._robot_state == RobotDriveState.TURN:
            
            p_trans, p_rot = 0.1 * self._wheel_max_speed_radps, 0.3 * self._wheel_max_speed_radps
            vl = p_trans * rho - p_rot * alpha 
            vr = p_trans * rho + p_rot * alpha

            if abs(alpha) < self._MIN_HEADING_ERROR:
                self._robot_state = RobotDriveState.DRIVE

        else:
            vl, vr = 0.0, 0.0


        vl = max(min(vl, self._wheel_max_speed_radps), -self._wheel_max_speed_radps)
        vr = max(min(vr, self._wheel_max_speed_radps), -self._wheel_max_speed_radps)

        return vl, vr
    
    def completed(self) -> bool:
        '''
            Controller has traversed all the waypoints provided to it. 
        '''
        if self._index >= len(self._waypoints):
            return True
        else:
            return False 
        
    def get_index(self) -> int:
        return self._index
    
    def get_current_target(self) -> float:
        return self._waypoints[self._index]