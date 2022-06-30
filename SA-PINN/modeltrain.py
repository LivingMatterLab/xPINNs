import tensorflow.compat.v1 as tf
import numpy as np
import time
import scipy.io as io

lr = 1.0e-3

class Train:
    def __init__(self, train_dict, n_max):
        self.train_dict = train_dict
        self.n_max = n_max
        '''
        self.sess = sess
        self.u_pred = u_pred
        self.loss = loss
        self.train_adam = train_adam
        self.train_lbfgs = train_lbfgs
        '''
        self.step = 0

    def callback(self, loss_):
        self.step += 1
        if self.step%1000 == 0:
            print('Loss: %.3e'%(loss_))


    #def nntrain(self, sess, u_pred, loss, train_adam, train_lbfgs, train_r, train_d, lambda_r, lambda_d, mu, k, a):
    def nntrain(self, sess, u_pred, loss, loss_f, loss_u, train_adam, train_r, train_d, lambda_r, lambda_d, mu, k, a):
        n = 0
        nmax = 50000
        nmax = self.n_max
        loss_c = 1.0e-3
        loss_ = 1.0
        Var_r = []
        Var_d = []
        Loss = []
        start_time = time.perf_counter()
        while n < nmax:# and loss_ > loss_c:
            n += 1
            if n <= 5000:
                u_, loss_, loss_f_, loss_u_, _, _ = sess.run([u_pred, loss, loss_f, loss_u, train_adam, train_r], feed_dict=self.train_dict)
            else:
                u_, loss_, loss_f_, loss_u_, _ = sess.run([u_pred, loss, loss_f, loss_u, train_adam], feed_dict=self.train_dict)

            lambda_r_, lambda_d_, mu_, k_, a_ = sess.run([lambda_r, lambda_d, mu, k, a])
            if n%10000 == 0:
                '''
                stop_time = time.perf_counter()
                time_sec = stop_time - start_time
                print('Time every 10 iterations: %.3f'%(time_sec))
                '''
                #check the variance for residual losses
                var_r = np.var(loss_f_)
                #check the variance for data loss
                var_d = np.var(loss_u_)
                Var_r.append(var_r)
                Var_d.append(var_d)
                Loss.append(loss_)
                print('Steps: %d, w_r: %.3e, w_d: %.3e, loss: %.3e'%(n, lambda_r_, lambda_d_, loss_))
                print('mu: %.3e, k: %.3e, a: %.3e, var_r: %.3e, var_d: %.3e'%(mu_, k_, a_, var_r, var_d))
                '''
                print('Steps: %d, loss: %.3e'%(n, loss_))
                '''
                #start_time = time.perf_counter()
        var_r_dict = np.asarray(Var_r)
        var_d_dict = np.asarray(Var_d)
        var_dict = {'var_r': var_r_dict, 'var_d': var_d_dict}
        io.savemat('./Output/var.mat', var_dict)
        loss_dict = np.asarray(Loss)
        loss_save = {'loss': loss_dict}
        io.savemat('./Output/loss.mat', loss_save)
        #train_lbfgs.minimize(sess, feed_dict=self.train_dict, fetches=[loss], loss_callback=self.callback)
