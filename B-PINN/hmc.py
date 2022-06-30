import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


class AdaptiveHMC:
    def __init__(
        self,
        target_log_prob_fn,
        init_state,
        num_results=1000,
        num_burnin=1000,
        num_leapfrog_steps=30,
        step_size=0.1,
    ):
        self.target_log_prob_fn = target_log_prob_fn
        self.init_state = init_state
        self.kernel = tfp.mcmc.SimpleStepSizeAdaptation(
            inner_kernel=tfp.mcmc.HamiltonianMonteCarlo(
                target_log_prob_fn=self.target_log_prob_fn,
                num_leapfrog_steps=num_leapfrog_steps,
                step_size=step_size,
            ),
            num_adaptation_steps=int(0.8 * num_burnin),
            target_accept_prob=0.75,
        )
        self.num_results = num_results
        self.num_burnin = num_burnin

    @tf.function
    def run_chain(self):
        samples, results = tfp.mcmc.sample_chain(
            num_results=self.num_results,
            num_burnin_steps=self.num_burnin,
            current_state=self.init_state,
            kernel=self.kernel,
        )
        return samples, results
