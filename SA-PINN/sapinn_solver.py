'''
SA-PINN for Covid-19 Forecast
@Author: Xuhui Meng (Division of Applied Mathematics, Brown University)
@Email: xuhui_meng@brown.edu
'''
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
#from tensorflow.contrib.opt import ScipyOptimizerInterface
import numpy as np
import time
import matplotlib.pyplot as plt

from dataset import Dataset
from net import DNN
from modeltrain import Train
from saveplot import SavePlot
from sklearn.metrics import mean_squared_error



wd = os.path.abspath(os.getcwd())
filename = os.path.basename(__file__)[:-3]

path2saveResults = 'Results/'+filename
if not os.path.exists(path2saveResults):
    os.makedirs(path2saveResults)
    
    
    
# nmax = 2000 b
#%% Plotting loading
ColorS = [0.5, 0.00, 0.0]
ColorE = [0.8, 0.00, 0.0]
ColorI = [1.0, 0.65, 0.0]
ColorR = [0.0, 0.00, 0.7]

#%% load data

covid_world = np.loadtxt("Data/covid_world.dat")

# make time array
days = np.arange(0, covid_world.shape[0])
d1=350
d2=700


#%%


np.random.seed(1234)
tf.set_random_seed(1234)

#size of the DNN
layers = [1] + 2*[32] + [1]

N_d_train = 225
nmax = 50000
# nmax = 200000


    

def plotting(t, u, t_r, u_all, u_pred, N_d_train):
    plt.figure(figsize=(1100/72,400/72))
    
    plt.scatter(t[:N_d_train], u[:N_d_train], s=300,edgecolors=(0.0, 0.0, 0.7),marker='.',facecolors='none', lw=1.0,zorder=1, alpha=1.0, label=r'train data') 
    
    plt.plot(t_r, u_all,linewidth=3.0,color=ColorI, label='smoothed daily new cases')
    plt.plot((days[d1:d2]-d1)/365, covid_world[d1:d2,1]/1e06,color=ColorI, lw=1.0, alpha=1.0,label='daily new cases')
    
    
    plt.plot(t_r, u_pred, color=(0.7, 0.0, 0.0),linewidth=3.0,linestyle="--", zorder=3, alpha=1.0, label=r'SA-PINN new confirmed cases')
    # plt.legend(loc='lower left',fontsize=14,ncol=2, fancybox=True, framealpha=0.)
    # plt.title('SA-PINN with {} observations'.format(str(N_d_train)))
    # plt.xlim([0, 1.0])
    plt.ylim([0.25,1])
    plt.xlabel("time [years]",fontsize=22)
    plt.ylabel("cases [m]",fontsize=22)
    plt.tight_layout()
    plt.savefig(path2saveResults+'/SA-PINN_Covid_data_'+str(N_d_train)+'_v1.pdf') 
    plt.show()



    
def main(N_d_train,nmax):
    # N_d_train = 225

    data = Dataset(N_d_train)
    #inputdata
    # t, u, t_r, tmin, tmax = data.build_data()
    t, u, t_r, u_all, tmin, tmax = data.build_data()
    N_r_train = t_r.shape[0]

    t_d_train = tf.placeholder(shape=[None, 1], dtype=tf.float32)
    u_d_train = tf.placeholder(shape=[None, 1], dtype=tf.float32)
    t_train = tf.placeholder(shape=[None, 1], dtype=tf.float32)
    
    #physics-infromed neural networks
    pinn = DNN(layers, tmin, tmax, N_d_train, N_r_train)
    W, b = pinn.hyper_initial()
    
    
    mu = tf.Variable(2.2, dtype=tf.float32)
    k = tf.Variable(350.0, dtype=tf.float32)
    a = tf.Variable(0.56, dtype=tf.float32)


    alpha_r, alpha_d = pinn.loss_weight() 
    u_pred = pinn.fnn(t_d_train, W, b)
    f_pred = pinn.pdenn(t_train, W, b, mu, k, a)

    var_list = [W, b, mu, k, a]
    var_w = [alpha_r, alpha_d]

    lambda_r = alpha_r
    lambda_d = alpha_d

    alpha_r_m = tf.reduce_mean(lambda_r)
    alpha_d_m = tf.reduce_mean(lambda_d)

    loss_f = tf.square(lambda_r)*tf.square(f_pred)
    loss_u = tf.square(lambda_d)*tf.square(u_d_train - u_pred)




    loss = tf.reduce_mean(loss_f) + tf.reduce_mean(loss_u)

    train_adam = tf.train.AdamOptimizer(learning_rate=1.0e-3, beta1=0.99).minimize(loss, var_list=var_list)

    #weights for the residuals
    opt_r = tf.train.AdamOptimizer(learning_rate=1.0e-3, beta1=0.99)
    grads_vars_r = opt_r.compute_gradients(loss, [alpha_r])
    capped_grads_vars_r = [(-gv[0], gv[1]) for gv in grads_vars_r]
    train_r = opt_r.apply_gradients(capped_grads_vars_r)

    #weights for the training data, which are not used in this case
    opt_d = tf.train.AdamOptimizer(learning_rate=1.0e-3, beta1=0.99)
    grads_vars_d = opt_d.compute_gradients(loss, [alpha_d])
    capped_grads_vars_d = [(-gv[0], gv[1]) for gv in grads_vars_d]
    train_d = opt_d.apply_gradients(capped_grads_vars_d)

    '''
    train_lbfgs = ScipyOptimizerInterface(loss,
                                          var_list=W+b,
                                          method = "L-BFGS-B",
                                          options = {'maxiter': 50000,
                                                     'ftol': 1.0*np.finfo(float).eps
                                                    }
                                         )
    '''


    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    train_dict = {t_train: t_r, t_d_train: t, u_d_train: u}
    
    Model = Train(train_dict, n_max=nmax)
    start_time = time.perf_counter()
    #Model.nntrain(sess, u_pred, loss, train_adam, train_lbfgs, train_r, train_d, alpha_r_m, alpha_d_m, mu, k, a)
    Model.nntrain(sess, u_pred, loss, loss_f, loss_u, train_adam, train_r, train_d, alpha_r_m, alpha_d_m, mu, k, a)
    stop_time = time.perf_counter()
    print('Duration time is %.3f seconds'%(stop_time - start_time))

    NT_test = t_r.shape[0]
    datasave = SavePlot(sess, t_r, NT_test, lambda_r, lambda_d)
    datasave.saveplt(u_pred, t_d_train)

    test_dict = {t_d_train: t_r}
    u_test = sess.run(u_pred, feed_dict=test_dict)
    lambda_d_ = sess.run(lambda_d)
    lambda_r_ = sess.run(lambda_r)

    return t, u, t_r, u_all, u_test, lambda_d_, lambda_r_



if __name__ == '__main__':
    
    t, u, t_r, u_all, u_pred, lambda_d_, lambda_r_ = main(N_d_train, nmax)
    plotting(t, u, t_r, u_all, u_pred, N_d_train)




    

