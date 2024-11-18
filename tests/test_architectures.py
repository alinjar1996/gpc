import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
from flax import nnx

from gpc.architectures import (
    MLP,
    # ActionSequenceMLP,
    # DenoisingMLP,
    # print_module_summary,
)


def test_mlp_construction() -> None:
    """Create a simple MLP and verify sizes."""
    batch_size = 10
    input_size = 2
    output_size = 3

    # Create the model
    model = MLP([input_size, 128, 32, output_size], nnx.Rngs(0))

    # Make sure the model is constructed correctly
    input = jnp.zeros((batch_size, input_size))
    output = model(input)
    assert output.shape == (batch_size, output_size)

    # Print a summary of the model
    nnx.display(model)


def test_mlp_save_load() -> None:
    """Verify that we can pickle an MLP."""
    rng = jax.random.key(0)
    mlp = MLP(layer_sizes=(2, 3, 4))
    dummy_input = jnp.ones((3,))
    params = mlp.init(rng, dummy_input)

    original_output = mlp.apply(params, dummy_input)

    # Create a temporary path for saving stuff
    local_dir = Path("_test_mlp")
    local_dir.mkdir(parents=True, exist_ok=True)

    # Save the MLP
    model_path = local_dir / "mlp.pkl"
    with Path(model_path).open("wb") as f:
        pickle.dump(mlp, f)

    # Save the parameters (weights)
    params_path = local_dir / "params.pkl"
    with Path(params_path).open("wb") as f:
        pickle.dump(params, f)

    # Load the MLP and parameters
    with Path(model_path).open("rb") as f:
        new_mlp = pickle.load(f)
    with Path(params_path).open("rb") as f:
        new_params = pickle.load(f)

    # Check that the loaded MLP gives the same output
    new_output = new_mlp.apply(new_params, dummy_input)
    assert jnp.allclose(original_output, new_output)

    # Remove the temporary directory
    for p in local_dir.iterdir():
        p.unlink()
    local_dir.rmdir()


def test_action_sequence_mlp() -> None:
    """Test the trajectory-generation MLP."""
    num_steps = 5
    action_dim = 3
    net = ActionSequenceMLP(
        hidden_layers=(32, 32), num_steps=num_steps, action_dim=action_dim
    )

    rng = jax.random.key(0)
    rng, init_rng = jax.random.split(rng)
    dummy_input = jnp.ones(7)
    params = net.init(init_rng, dummy_input)

    U = net.apply(params, jnp.ones(7))
    assert U.shape == (num_steps, action_dim)

    U = net.apply(params, jnp.ones((14, 24, 7)))
    assert U.shape == (14, 24, num_steps, action_dim)


def test_denoising_mlp() -> None:
    """Test the denoising MLP."""
    rng = jax.random.key(0)
    num_steps = 5
    action_dim = 3
    obs_dim = 4

    # Define the network architecture
    net = DenoisingMLP(hidden_layers=(32, 32))

    # Initialize network parameters
    U = jnp.ones((num_steps, action_dim))
    y = jnp.ones(obs_dim)
    t = jnp.ones(1)
    params = net.init(rng, U, y, t)

    # Check the output shape
    U_out = net.apply(params, U, y, t)
    assert U_out.shape == (num_steps, action_dim)

    # Try with a batch of sequences
    U = jnp.ones((14, 24, num_steps, action_dim))
    y = jnp.ones((14, 24, obs_dim))
    t = jnp.ones((14, 24, 1))
    U_out = net.apply(params, U, y, t)
    assert U_out.shape == U.shape


if __name__ == "__main__":
    test_mlp_construction()
    # test_mlp_save_load()
    # test_action_sequence_mlp()
    # test_denoising_mlp()
