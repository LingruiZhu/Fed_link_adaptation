import numpy as np
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
import h5py

from MovingSensor import MovingSensor
from ChannelModel import ChannelModel, createMap
from Devices import RandomMovingUserEquipment, StaticUserEquipment, BaseStation
from MovingArea import MovingArea


def generate_sinr_data():
    # initialze the moving area
    x_boundary = np.array([0, 20])
    y_boundary = np.array([0, 20])
    moving_area = MovingArea(x_boundary, y_boundary)
    
    # initialize the randomly moving sensors
    interference_sensor_1 = RandomMovingUserEquipment(tx_power=20, init_x=5, init_y=5, speed=2, direction=0, x_boundary=x_boundary, y_boundary=y_boundary)
    interference_sensor_2 = RandomMovingUserEquipment(tx_power=20, init_x=10, init_y=10, speed=2, direction=0, x_boundary=x_boundary, y_boundary=y_boundary)
    interference_sensor_3 = RandomMovingUserEquipment(tx_power=20, init_x=10, init_y=10, speed=2, direction=0, x_boundary=x_boundary, y_boundary=y_boundary)
    
    # define the UE position (dont move)
    UE_1 = StaticUserEquipment(tx_power=0, pos_x=25, pos_y=10)
    
    # define Base station
    base_station = BaseStation(tx_power=1000, pos_x=30, pos_y=15)
    
    # initialize the channel (unified)
    map_sigma = 3
    map_delta = 5
    map_length = 40
    map_width = 40
    map_step_size = 1/20
    shadowing_map = createMap(map_width, map_length, map_sigma, map_delta, map_step_size)
    
    channel_1 = ChannelModel(path_loss_factor=2.5, number_paths=10, ue_speed=2, carrier_freq=3e9, shadowing_map=shadowing_map,\
        delta=map_delta, step_size=map_step_size, map_width=map_width, map_length=map_length)
    channel_2 = ChannelModel(path_loss_factor=2.5, number_paths=10, ue_speed=2, carrier_freq=3e9, shadowing_map=shadowing_map,\
        delta=map_delta, step_size=map_step_size, map_width=map_width, map_length=map_length)
    channel_3 = ChannelModel(path_loss_factor=2.5, number_paths=10, ue_speed=2, carrier_freq=3e9, shadowing_map=shadowing_map,\
        delta=map_delta, step_size=map_step_size, map_width=map_width, map_length=map_length)
    channel_BS = ChannelModel(path_loss_factor=2.5, number_paths=10, ue_speed=2, carrier_freq=3e9, shadowing_map=shadowing_map,\
        delta=map_delta, step_size=map_step_size, map_width=map_width, map_length=map_length)
    
    # write the loop for updating postion of sensors and calculate SINR
    num_samples = 10000
    sample_frequency = 1000
    sample_interval = 1 / sample_frequency
    durarion = num_samples * sample_interval
    
    time_indicies = np.linspace(0, durarion, num_samples)
    
    UE_position = np.array([UE_1.position_x, UE_1.position_y])
    BS_position = np.array([base_station.position_x, base_station.position_y])
    
    sinr_list = list()
    sinr_dB_list = list()
    interference_list = list()
    for time_index in tqdm(time_indicies):
        inter1_position = interference_sensor_1.random_move(sampling_interval=sample_interval)
        inter2_position = interference_sensor_2.random_move(sampling_interval=sample_interval)
        inter3_position = interference_sensor_3.random_move(sampling_interval=sample_interval)
        
        # calculate interfernce
        interference_channel_1 = channel_1.calculate_channel(UE_position, inter1_position, time_index)
        interference_channel_2 = channel_2.calculate_channel(UE_position, inter2_position, time_index)
        interference_channel_3 = channel_3.calculate_channel(UE_position, inter3_position, time_index)
        interference_power = interference_channel_1 * interference_sensor_1.tx_power + \
            interference_channel_2 * interference_sensor_2.tx_power + interference_channel_3 * interference_sensor_3.tx_power
        
        noise_power = 0.01
        
        interference_list.append(interference_power)
        signal_power = base_station.tx_power * channel_BS.calculate_channel(UE_position, BS_position, time_index)
        sinr = signal_power / (noise_power + interference_power)
        sinr_dB = 10*math.log10(sinr)
        sinr_list.append(sinr)
        sinr_dB_list.append(sinr_dB)
    
    plt.figure()
    plt.plot(time_indicies, sinr_list)
    plt.xlabel("time")
    plt.ylabel("SINR")
    plt.grid()
    
    plt.figure()
    plt.plot(time_indicies, sinr_dB_list)
    plt.xlabel("time")
    plt.ylabel("SINR(in dB)")
    plt.grid()
    plt.show()    
    
    file_single_ue = h5py.File("Interference_generation/single_UE_data.h5", "w")
    file_single_ue.create_dataset(name="SINR", data=np.array(sinr_list))
    file_single_ue.create_dataset(name="SINR_dB", data=np.array(sinr_dB_list))
    file_single_ue.create_dataset(name="Interference_power", data=np.array(interference_list))
    file_single_ue.close()


if __name__ == "__main__":
    generate_sinr_data()
    