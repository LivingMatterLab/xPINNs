import tensorflow as tf
import numpy as np

class DNN:
    def __init__(self, layer_size, Xmin, Xmax, N_d, N_r):
        self.size = layer_size
        self.Xmin = Xmin
        self.Xmax = Xmax
        self.N_r = N_r
        self.N_d = N_d
    
    def hyper_initial(self):
        L = len(self.size)
        Weights = []
        Biases = []
        for l in range(1, L):
            in_dim = self.size[l-1]
            out_dim = self.size[l]
            std = np.sqrt(2/(in_dim + out_dim))
            weight = tf.Variable(tf.random.normal(shape=[in_dim, out_dim], stddev=std))
            bias = tf.Variable(tf.zeros(shape=[1, out_dim]))
            Weights.append(weight)
            Biases.append(bias)

        return Weights, Biases

    def loss_weight(self):
        '''
        alpha_r = tf.Variable(tf.ones(shape=[self.N_r, 1]), dtype=tf.float32)
        alpha_d = tf.Variable(tf.ones(shape=[self.N_d, 1]), dtype=tf.float32)
        '''
        alpha_r = tf.Variable(tf.random.uniform(shape=[self.N_r, 1]), dtype=tf.float32)
        alpha_d = tf.Variable(tf.random.uniform(shape=[self.N_d, 1]), dtype=tf.float32)
        return alpha_r, alpha_d

    def fnn(self, X, W, b):
        A = 2.0*(X - self.Xmin)/(self.Xmax - self.Xmin) - 1.0
        #A = X
        L = len(W)
        for i in range(L-1):
            A = tf.tanh(tf.add(tf.matmul(A, W[i]), b[i]))
        Y = tf.add(tf.matmul(A, W[-1]), b[-1])
        
        return Y

    def pdenn(self, t, W, b, mu, k, a):
        u = self.fnn(t, W, b)
        u_t = tf.gradients(u, t)[0]
        u_tt = tf.gradients(u_t, t)[0]
        m = 1.0e-3
        f = m*(u_tt + mu*u_t + k*(u - a))





        return f
