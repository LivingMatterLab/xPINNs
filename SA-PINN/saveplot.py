import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
'''
savepath='./Output'
if not os.path.exists(savepath):
    os.makedirs(savepath)
'''

class SavePlot:
    def __init__(self, sess, t_range, NT, lambda_r, lambda_d):
        self.t_range = t_range
        self.NT = NT
        self.sess = sess
        self.lambda_r = lambda_r
        self.lambda_d = lambda_d

    def saveplt(self, u_pred, t_d_train):
        t_test = self.t_range
        test_dict = {t_d_train: t_test}
        u_test = self.sess.run(u_pred, feed_dict=test_dict)
        np.savetxt('./Output/u_pred', u_test, fmt='%e')

        np.savetxt('./Output/t', t_test, fmt='%e')
        lambda_r_ = self.sess.run(self.lambda_r)
        np.savetxt('./Output/w_r', lambda_r_, fmt='%e')
        lambda_d_ = self.sess.run(self.lambda_d)
        np.savetxt('./Output/w_d', lambda_d_, fmt='%e')

        '''
        plt.imshow(u_test, cmap='rainbow', aspect='auto')
        plt.colorbar()
        plt.savefig('./Output/ucontour.png')
        plt.show()
        '''
