import tensorflow as tf
import tensorflow_probability as tfp


tfd = tfp.distributions


class PI_Bayesian:
    def __init__(
        self,
        x_u,
        y_u,
        x_pde,
        pde_fn,
        L=6,
        noise_u=0.05,
        noise_pde=0.05,
        prior_sigma=1.0,
    ):
        self.x_u = x_u
        self.y_u = y_u
        self.x_pde = x_pde

        self.pde_fn = pde_fn
        self.L = L
        self.noise_u = noise_u
        self.noise_pde = noise_pde
        self.prior_sigma = prior_sigma

        self.log_mu_init = tf.math.log(2.2)
        self.log_k_init = tf.math.log(350.0)
        self.log_b_init = tf.math.log(0.56)
        
        # self.log_mu_init = tf.math.log(-5.0)
        # self.log_k_init = tf.math.log(1.0)
        # self.log_b_init = tf.math.log(0.56)
        
        self.additional_inits = [self.log_mu_init, self.log_k_init, self.log_b_init]
        self.additional_priors = [
            tfd.Normal(0, scale=0.5),
            tfd.Normal(0, scale=0.5),
            tfd.Normal(0, scale=0.5),
        ]

    def build_posterior(self, bnn_fn):
        y_u = tf.constant(self.y_u, dtype=tf.float32)

        def _fn(*variables):
            """
            log posterior function, which takes neural network's parameters input, and outputs (probably unnormalized) density probability
            """
            # split the input list into variables for neural networks, and additional variables
            variables_nn = variables[: 2 * self.L]
            log_mu, log_k, log_b = variables[2 * self.L :]
            mu, k, b = (
                tf.exp(log_mu + self.log_mu_init),
                tf.exp(log_k + self.log_k_init),
                tf.exp(log_b + self.log_b_init),
            )
            # explicitly create a tf.Tensor here, for input to neural networks, to avoid bugs
            x_u = tf.constant(self.x_u, dtype=tf.float32)
            x_pde = tf.constant(self.x_pde, dtype=tf.float32)

            # make inference
            _fn = lambda x: bnn_fn(x, variables_nn)
            y_u_pred = _fn(x_u)
            pde_pred = self.pde_fn(x_pde, _fn, [mu, k, b])

            # construct prior distributions, likelihood distributions
            u_likeli = tfd.Normal(loc=y_u, scale=self.noise_u * tf.ones_like(y_u))
            noise_pde1, noise_pde2 = self.noise_pde
            N1, N2 = y_u_pred.shape[0], pde_pred.shape[0]
            pde_likeli_1 = tfd.Normal(
                loc=tf.zeros([N1, 1]), scale=noise_pde1 * tf.ones([N1, 1])
            )
            pde_likeli_2 = tfd.Normal(
                loc=tf.zeros([N2 - N1, 1]), scale=noise_pde2 * tf.ones([N2 - N1, 1])
            )
            # pde_likeli = tfd.Normal(loc=tf.zeros_like(pde_pred), scale=self.noise_pde*tf.ones_like(pde_pred))

            prior = tfd.Normal(loc=0, scale=self.prior_sigma)

            # compute unnormalized log posterior, by adding log prior and log likelihood
            log_prior = tf.reduce_sum(
                [tf.reduce_sum(prior.log_prob(var)) for var in variables_nn]
            ) + tf.reduce_sum(
                [
                    dist.log_prob(v)
                    for v, dist in zip([log_mu, log_k, log_b], self.additional_priors)
                ]
            )
            # log_prior += tf.reduce_sum([dist.log_prob(v) for v, dist in zip([(mu-2.2)/2.2, (k-370.0)/370.0, (b-0.56)/0.56], self.additional_priors)])
            log_likeli = (
                tf.reduce_sum(u_likeli.log_prob(y_u_pred))
                + tf.reduce_sum(pde_likeli_1.log_prob(pde_pred[:N1, :]))
                + tf.reduce_sum(pde_likeli_2.log_prob(pde_pred[N1:N2, :]))
            )
            return log_prior + log_likeli

        return _fn
