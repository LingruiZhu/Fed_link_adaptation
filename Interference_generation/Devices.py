import numpy as np
import math
import random
import matplotlib.pyplot as plt

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
            

class RouteMovingUserEquipment:
    def __init__(self, tx_power, init_x, init_y, speed_mps, route_type:str, direction=0, sampling_interval:float=0.001, radius:int=10,\
        x_boundary=[0, 20], y_boundary=[0, 20]):
        self.tx_power = tx_power
        self.position_x = init_x
        self.position_y = init_y
        self.speed = speed_mps
        self.direction = direction
        self.x_boundary = x_boundary
        self.y_boundary = y_boundary
        self.radius = radius
        self.route_type = route_type
        self.rad_speed = speed_mps / self.radius
        self.sampling_interval = sampling_interval
        theta = sampling_interval * self.rad_speed
        self.circle_motion_matrix = np.array([[math.cos(theta), -math.sin(theta)],
                                              [math.sin(theta), math.cos(theta)]])

    def move(self):
        if self.route_type == "circle":
            position = self.move_in_circle()
        elif self.route_type == "square":
            position = self.move_in_square()
        else:
            raise ValueError("Invalid route_type. Supported types: 'circle' or 'square'.")
        return position
        
        
    def move_in_circle(self):
        self.direction = random.uniform(-math.pi, +math.pi)
        old_position = np.array([self.position_x, self.position_y]).transpose()
        new_position = np.matmul(self.circle_motion_matrix, old_position)
        self.position_x, self.position_y = new_position[0], new_position[1]
        self.check_position()
        return np.array([self.position_x, self.position_y])
    

    def move_in_square(self):
        x1, y1, x2, y2 = self.x_boundary[0], self.y_boundary[0], self.x_boundary[1], self.y_boundary[1]
        step = self.speed * self.sampling_interval

        if x1 <= self.position_x < x2 and self.position_y == y2:        # upper bound 
            self.position_x += step
        elif x1 < self.position_x <= x2 and self.position_y == y1:
            self.position_x -= step
        elif self.position_x == x1 and y1 <= self.position_y < y2:
            self.position_y += step
        elif self.position_x == x2 and y1 < self.position_y <= y2:
            self.position_y -= step
        
        if self.position_x > x2:
            self.postion_x = x2
        elif self.position_x < x1:
            self.position_x = x1
        
        if self.position_y > y2:
            self.position_y = y2
        elif self.position_y < y1:
            self.position_y = y1
        return np.array([self.position_x, self.position_y])

    def check_position(self):
        # Add logic to ensure that the position stays within the specified boundaries (x_boundary, y_boundary).
        pass

    def get_position(self):
        return np.array([self.position_x, self.position_y])
        
            
        
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
        

if __name__ == "__main__":
    # Create an instance of RouteMovingUserEquipment
    user_equipment = RouteMovingUserEquipment(tx_power=1, init_x=0, init_y=10, radius=10, speed_mps=2, x_boundary=[0, 20], y_boundary=[0, 20])

    # Define the total number of steps
    total_steps = 100000

    # Lists to store the x and y coordinates for plotting
    x_coordinates = [user_equipment.get_position()[0]]
    y_coordinates = [user_equipment.get_position()[1]]

    # Move in a circle
    for _ in range(total_steps):
        user_equipment.move("circle")  # Adjust the sampling interval as needed
        x_coordinates.append(user_equipment.get_position()[0])
        y_coordinates.append(user_equipment.get_position()[1])

    # Plot the path
    plt.figure(figsize=(6, 6))
    plt.plot(x_coordinates, y_coordinates, marker='o', linestyle='-', color='b')
    plt.title("Path of Route Moving User Equipment (Circle)")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True)
    plt.show()

    # Reset the user equipment
    user_equipment = RouteMovingUserEquipment(tx_power=1, init_x=20, init_y=20, speed_mps=2, sampling_interval=0.001, x_boundary=[0, 20], y_boundary=[0, 20])

    # Lists to store the x and y coordinates for plotting
    x_coordinates = [user_equipment.get_position()[0]]
    y_coordinates = [user_equipment.get_position()[1]]

    # Move in a square
    for _ in range(total_steps):
        user_equipment.move("square")  # Adjust the sampling interval as needed
        x_coordinates.append(user_equipment.get_position()[0])
        y_coordinates.append(user_equipment.get_position()[1])

    # Plot the path
    plt.figure(figsize=(6, 6))
    plt.plot(x_coordinates, y_coordinates, marker='o', linestyle='-', color='g')
    plt.title("Path of Route Moving User Equipment (Square)")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True)
    plt.show()
