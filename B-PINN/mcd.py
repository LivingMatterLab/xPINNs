import tensorflow as tf
import tensorflow_probability as tfp


class MCD(tf.keras.Model):
    """Monte Carlo dropout for UQ"""

    def __init__(self, hidden_units=50, dropout_rate_u=0.005, dropout_rate_ode=0.001):
        super().__init__()
        self.denses = [
            tf.keras.layers.Dense(hidden_units, activation=tf.tanh),
            tf.keras.layers.Dense(hidden_units, activation=tf.tanh),
            tf.keras.layers.Dense(1),
        ]
        self.dropout_rate_u = dropout_rate_u
        self.dropout_rate_ode = dropout_rate_ode

        self.log_mu = tf.Variable(tf.math.log(2.2), dtype=tf.float32)
        self.log_k = tf.Variable(tf.math.log(350.0), dtype=tf.float32)
        self.log_b = tf.Variable(tf.math.log(0.56), dtype=tf.float32)

        self.opt = tf.keras.optimizers.Adam(learning_rate=1e-3)

    def call(self, t, dropout_rate=0.0):
        dropout_layer = tf.keras.layers.Dropout(dropout_rate)
        y = t
        for i in range(len(self.denses) - 1):
            y = self.denses[i](y)
            y = dropout_layer(y, training=True)
        return self.denses[-1](y)

    def ODE(self, t, dropout_rate=0.0):
        with tf.GradientTape() as g_tt:
            g_tt.watch(t)
            with tf.GradientTape() as g_t:
                g_t.watch(t)
                x = self.call(t, dropout_rate=dropout_rate)
            x_t = g_t.gradient(x, t)
        x_tt = g_tt.gradient(x_t, t)
        return (
            1 / tf.exp(self.log_k) * x_tt
            + tf.exp(self.log_mu) / tf.exp(self.log_k) * x_t
            + (x - tf.exp(self.log_b))
        )

    def train_op(self, t_u, x_u, t_ode):
        with tf.GradientTape() as tape:
            x_u_pred = self.call(t_u, dropout_rate=self.dropout_rate_u)
            f_ode = self.ODE(t_ode, dropout_rate=self.dropout_rate_ode)
            loss = tf.reduce_mean((x_u_pred - x_u) ** 2) + tf.reduce_mean(f_ode**2)
        grads = tape.gradient(loss, self.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.trainable_variables))
        return loss

    def train(self, t_u, x_u, t_ode, niter=10000):
        train_op = tf.function(self.train_op)
        loss = []
        for it in range(niter):
            loss_value = train_op(t_u, x_u, t_ode)
            loss += [loss_value.numpy()]
            if it % 100000 == 0:
                print(it, loss[-1])
        return loss

    def infer(self, t, num_samples=1000):
        tt = tf.tile(t, [num_samples, 1])
        x = self.call(tt, dropout_rate=self.dropout_rate_u)
        return tf.reshape(x, [num_samples, t.shape[0], -1])
