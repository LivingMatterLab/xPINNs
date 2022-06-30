import numpy as np
import tensorflow as tf


class BNN:
    def __init__(self, layers, activation=tf.tanh):
        self.L = len(layers) - 1
        self.variables = self.init_network(layers)
        self.bnn_fn = self.build_bnn()
        self.bnn_infer_fn = self.build_infer()
        self.activation = activation

    def init_network(self, layers):
        W, b = [], []
        init = tf.zeros
        # init = tf.keras.initializers.glorot_normal()
        for i in range(self.L):
            W += [init(shape=[layers[i], layers[i + 1]], dtype=tf.float32)]
            b += [tf.zeros(shape=[1, layers[i + 1]], dtype=tf.float32)]
        return W + b

    def build_bnn(self):
        def _fn(x, variables):
            """
            BNN function, for one realization of the neural network, used for MCMC

            Args:
            -----
            x: input,
                tensor, with shape [None, input_dim]
            variables: weights and bias,
                list of tensors, each one of which has dimension [:, :]

            Returns:
            --------
            y: output,
                tensor, with shape [None, output_dim]
            """
            W = variables[: len(variables) // 2]
            b = variables[len(variables) // 2 :]
            y = x
            for i in range(self.L - 1):
                y = self.activation(tf.matmul(y, W[i]) + b[i])
            return tf.matmul(y, W[-1]) + b[-1]

        return _fn

    def build_infer(self):
        def _fn(x, variables):
            """
            BNN function, for batch of realizations of the neural network, used for inference

            Args:
            -----
            x: input,
                tensor, with shape [None, input_dim]
            variables: weights and bias,
                list of tensors, each one of which has dimension [batch_size, :, :]

            Returns:
            --------
            y: output,
                tensor, with shape [batch_size, None, output_dim]
            """
            W = variables[: len(variables) // 2]
            b = variables[len(variables) // 2 :]
            batch_size = W[0].shape[0]
            y = tf.tile(x[None, :, :], [batch_size, 1, 1])
            for i in range(self.L - 1):
                y = self.activation(tf.einsum("Nij,Njk->Nik", y, W[i]) + b[i])
            return tf.einsum("Nij,Njk->Nik", y, W[-1]) + b[-1]

        return _fn
