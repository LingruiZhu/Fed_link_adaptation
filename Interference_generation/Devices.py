import numpy as np
import math
import random

class RandomMovingUserEquipment:
    def __init__(self, tx_power, init_x, init_y, speed, direction=0, x_boundary:list=[0, 20], y_boundary:list=[0,20]) -> None:
        self.tx_power = tx_power
        self.position_x = init_x
        self.position_y = init_y
        self.speed = speed
        self.direction = direction
        self.x_boundary = x_boundary
        self.y_boundary = y_boundary
    
    
    def random_move(self, sampling_interval):
        self.direction = random.uniform(-math.pi, +math.pi)
        step_x = self.speed * sampling_interval * math.cos(self.direction)
        step_y = self.speed * sampling_interval * math.sin(self.direction)
        self.position_x += step_x
        self.position_y += step_y
        self.check_position()
        return np.array([self.position_x, self.position_y])
    
    
    def check_position(self):
        if self.position_x <= self.x_boundary[0]:
            self.position_x = self.x_boundary[0]
        elif self.position_x >= self.x_boundary[1]:
            self.position_x = self.x_boundary[1]
        
        if self.position_y <= self.y_boundary[0]:
            self.position_y = self.y_boundary[0]
        elif self.position_y >= self.y_boundary[1]:
            self.position_y = self.y_boundary[1]
            
        
class StaticUserEquipment:
    def __init__(self, tx_power, pos_x, pos_y) -> None:
        self.tx_power = tx_power
        self.position_x = pos_x
        self.position_y = pos_y
        
        
class BaseStation:
    def __init__(self, tx_power, pos_x, pos_y) -> None:
        self.tx_power = tx_power
        self.position_x = pos_x
        self.position_y = pos_y