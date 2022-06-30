import numpy as np
import tensorflow as tf

DTYPE = "float32"
tf.keras.backend.set_floatx(DTYPE)


def load_covid_data(d1=350, d2=700):
    # world-wide COVID cases, daily, since start of pandemic
    covid_world = np.loadtxt("covid_world.dat")

    # make time array
    days = np.arange(0, covid_world.shape[0])
    years = days / 365

    def movingaverage(interval, window_size):
        window = np.ones(int(window_size)) / float(window_size)
        return np.convolve(interval, window, "same")

    # smooth data to take out weekly fulctuations due to reporting
    covid_world_smooth = movingaverage(covid_world[:, 1], 7)

    #### create training data arrays
    t_covid = days[d1:d2] - d1  # time array: days
    y_covid = t_covid / 365  # time array: years

    x_covid = (
        covid_world_smooth[d1:d2] / 1e06
    )  # normalize COVID numbers per 10^6 people

    # pick range of training data
    t_data = y_covid[0:225:1]
    x_data = x_covid[0:225:1]

    # collocation points for enforcing ODE (whole window of interest)
    t_physics = y_covid[0::1]

    # convert arrays to tf tensors
    t_data_tf = tf.convert_to_tensor(t_data, dtype=DTYPE)
    x_data_tf = tf.convert_to_tensor(x_data, dtype=DTYPE)
    x_data_un_tf = tf.convert_to_tensor(covid_world[d1:d2,1]/1e06, dtype=DTYPE)
    t_physics_tf = tf.convert_to_tensor(t_physics, dtype=DTYPE)

    T_data = tf.reshape(t_data_tf[:], shape=(t_data.shape[0], 1))
    X_data = tf.reshape(x_data_tf[:], shape=(x_data.shape[0], 1))
    
    X_data_un = tf.reshape(x_data_un_tf[:], shape=(x_covid.shape[0], 1))
    
    T_r = tf.reshape(t_physics_tf[:], shape=(t_physics.shape[0], 1))

    # pick the exact (smoothed) data
    T_exact = y_covid.copy()
    X_exact = x_covid.copy()

    return T_data, X_data, T_r, T_exact, X_exact, (days[d1:d2]-d1)/365, covid_world[d1:d2,1]/1e06, X_data_un



