import tensorflow as tf
import numpy as np
import scipy.io as io
import matplotlib.pyplot as plt

class Dataset:
    def __init__(self, N_train):
        self.N_train = N_train

    def window_average(self, interval, window_size):
        window = np.ones(int(window_size))/float(window_size)
        return np.convolve(interval, window, 'same')

    def build_data(self):
        covid_raw = np.loadtxt('./Data/covid_world.dat')
        covid_data = self.window_average(covid_raw[:, 1], 7)/1.0e6
        covid_data = np.reshape(covid_data, (-1, 1))
        days = np.arange(0, covid_raw.shape[0])
        years = days/365

        d1 = 350
        d2 = 700
        covid_window = covid_data[d1:d2, :]
        #covid_data = covid_raw[d1:d2, 1]/1.0e6

        t_r = (days[d1:d2] - d1)/365
        t_r = np.reshape(t_r, (-1, 1))
        tmin = t_r.min(0)
        tmax = t_r.max(0)
        t_d = t_r[:self.N_train, :]
        u_d = covid_window[:self.N_train, :]

        data_ref = {'t_train': t_d, 'u_train': u_d, 't': t_r, 'u': covid_window}
        io.savemat('./Output/data_ref.mat', data_ref)

        '''
        plt.figure()
        plt.plot(t_r, covid_window, 'k-')
        plt.plot(t_d, u_d, 'ro')
        plt.show()
        '''
        return t_d, u_d, t_r, covid_window, tmin, tmax
        # return t_d, u_d, t_r, tmin, tmax
