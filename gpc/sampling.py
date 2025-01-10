from hydrax.algs import PredictiveSampling

from gpc.policy import Policy


class BootstrappedPredictiveSampling(PredictiveSampling):
    """Perform predictive sampling, but add samples from a generative policy."""

    def __init__(self, policy: Policy, num_policy_samples: int, **kwargs):
        """Initialize the controller.

        Args:
            policy: The generative policy to sample from.
            num_policy_samples: The number of samples to take from the policy.
            **kwargs: Constructor arguments for PredictiveSampling.
        """
        self.policy = policy
        self.num_policy_samples = num_policy_samples
        super().__init__(**kwargs)
