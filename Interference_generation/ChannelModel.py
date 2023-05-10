import numpy as np
import matplotlib.pyplot as plt


def calculate_path_loss(x, y, alpha):
    '''
    non singulat distance dependent path loss is utilized :
    https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8697126    eq(4)
    loss exponent(alpha>2)
    '''
    return min(1, np.linalg.norm(y-x)**(-alpha))


def calculate_small_scale_fading(t, M, doppler, beta_n, theta):
    '''
    modified jakes fading is utilized
    reason: to generate uncorrelated fading waveforms.
    https://www.ee.iitm.ac.in/~giri/pdfs/EE5141/EE5141/Jakes-model-revisited.pdf eq:2
    '''
    H = np.abs(np.sqrt(2/M)*(np.sum((np.cos(beta_n)+1j*np.sin(beta_n))*np.cos(doppler*t+theta))))
    return H**2


def calculate_shadowing(grf, delta, loc1, loc2, stepsize, height, width):
    grf1 = grf[int((loc1[0])/stepsize)+1][int((loc1[1])/stepsize)+1]
    grf2 = grf[int((loc2[0])/stepsize)+1][int((loc2[1])/stepsize)+1]
    dist = np.linalg.norm(loc1-loc2)
    return 10**(((1-np.exp(-dist/delta))/(np.sqrt(2)*np.sqrt(1+np.exp(-dist/delta)))*(grf1+grf2))/10)


def createMap(width, height, sigmaS, correlationDistance, stepsize):
    '''
    Book :“Stochastic Geometry, Spatial Statistics and Random Fields:,”, page 374 
    '''
    num_x_points = int(width/stepsize) + 3
    num_y_points = int(height/stepsize) + 3
    mapXPoints=np.linspace(0, width, num=num_x_points, endpoint=True)
    mapYPoints=np.linspace(0, height, num=num_y_points, endpoint=True)
    
    N1 = len(mapXPoints)
    N2 = len(mapYPoints)
    G = np.zeros([N1,N2],dtype=np.float64)
    for n in range(N1):
        for m in range(N2):
            G[n,m]= sigmaS*np.exp(-1*np.sqrt(np.min([np.absolute(mapXPoints[0]-mapXPoints[n]),\
                                      width-np.absolute(mapXPoints[0]-mapXPoints[n])])**2\
            + np.min([np.absolute(mapYPoints[0]-mapYPoints[m]),height\
                  -np.absolute(mapYPoints[0]-mapYPoints[m])])**2)/correlationDistance)
    Gamma = np.fft.fft2(G)
    Z = np.random.randn(N1,N2) + 1j*np.random.randn(N1,N2)
    mapp = np.real(np.fft.fft2(np.multiply(np.sqrt(Gamma),Z)\
                               /np.sqrt(N1*N2)))
    return mapp


class ChannelModel:
    def __init__(self, path_loss_factor, number_paths, ue_speed, carrier_freq, shadowing_map, delta, step_size, map_width, map_length) -> None:
        self.path_loss_factor = path_loss_factor    # parameter for path loss
            
        self.number_paths = number_paths            # parameter for small scale fading
        self.n_nut = int(self.number_paths / 4)
        self.ue_speed = ue_speed                        # for doppler spread
        self.carrier_freq = carrier_freq
        self.light_speed = 299792458
        self.alpha_n = np.array([2*np.pi*(n-0.5)/self.number_paths for n in range(1,self.n_nut+1)])
        self.beta_n = [np.pi*n/(int(self.n_nut)) for n in range(1, self.n_nut+1)]
        self.doppler_spread = 2*np.pi * self.carrier_freq * self.ue_speed *np.cos(self.alpha_n) / self.light_speed
        self.theta = np.random.uniform(0, 2*np.pi, self.n_nut)
        
        self.shadowing_map = shadowing_map          # for shadowing
        self.map_width = map_width
        self.map_length = map_length
        self.delta = delta
        self.step_size = step_size                  # should be the resolution for shadowing map
        
        
    def calculate_channel(self, position1, position2, time_index):
        path_loss = calculate_path_loss(position1, position2, self.path_loss_factor)
        small_scale_fading = calculate_small_scale_fading(time_index, self.n_nut, self.doppler_spread, self.beta_n, self.theta)
        shadowing = calculate_shadowing(self.shadowing_map, self.delta, position1, position2, self.step_size, self.map_length, self.map_width)
        channel_power_attenuation = path_loss * small_scale_fading * shadowing
        return channel_power_attenuation


def channel_test():
    # parameter initialization
    path_loss_factor = 2
    number_paths = 20
    ue_speed = 1
    carrier_freq = 3e9
    
    # shadowing map initialization
    map_sigma = 3
    map_delta = 5
    map_length = 20
    map_width = 20
    map_step_size = 0.05
    shadowing_map = createMap(map_width, map_length, map_sigma, map_delta, map_step_size)
    
    # initialize channel model
    channel_model_1 = ChannelModel(path_loss_factor, number_paths, ue_speed, carrier_freq, shadowing_map, map_delta,\
        map_step_size, map_width, map_length)
    
    ue_speed = 3
    channel_model_2 = ChannelModel(path_loss_factor, number_paths, ue_speed, carrier_freq, shadowing_map, map_delta,\
        map_step_size, map_width, map_length)
    
    sampling_freq  = 1000
    time_duration_second = 1
    num_samples = int(time_duration_second * sampling_freq)
    time_indicies = np.linspace(0, time_duration_second, num_samples, endpoint=False)
    
    position1 = np.array([0,0])
    position2 = np.array([10, 10])
    
    channel_power_attenuation_sequence_1 = list()
    channel_power_attenuation_sequence_2 = list()
    for time_index in time_indicies:
        channel_power_attenuation_sequence_1.append(channel_model_1.calculate_channel(position1, position2, time_index))
        channel_power_attenuation_sequence_2.append(channel_model_2.calculate_channel(position1, position2, time_index))
    plt.figure()
    plt.plot(time_indicies, channel_power_attenuation_sequence_1, "r-x", label="UE_speed = 1 m/s")
    plt.plot(time_indicies, channel_power_attenuation_sequence_2, "b-x", label="Ue_speed = 3 m/s")
    plt.legend()
    plt.grid()
    plt.xlabel("Time")
    plt.ylabel("Channel Power Attenuation")
    plt.show()
    

if __name__ == "__main__":
    channel_test()
        
        