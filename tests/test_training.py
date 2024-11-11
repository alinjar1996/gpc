from hydrax.algs import PredictiveSampling

from gpc.particle import ParticleEnv
from gpc.training import train


def test_train() -> None:
    """Test the training loop."""
    env = ParticleEnv()
    ctrl = PredictiveSampling(env.task, num_samples=8, noise_level=0.1)
    train(env, ctrl)


if __name__ == "__main__":
    test_train()
