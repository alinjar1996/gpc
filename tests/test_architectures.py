import pickle
from pathlib import Path

import jax
import jax.numpy as jnp

from gpc.architectures import MLP, ActionSequenceMLP, print_module_summary


def test_mlp_construction() -> None:
    """Create a simple MLP and verify sizes."""
    input_size = (3,)
    layer_sizes = (2, 3, 4)

    # Pseudo-random keys
    rng = jax.random.key(0)
    rng, param_rng, input_rng = jax.random.split(rng, 3)

    # Create the MLP
    mlp = MLP(layer_sizes=layer_sizes, bias=True)
    dummy_input = jnp.ones(input_size)
    params = mlp.init(param_rng, dummy_input)

    # Check the MLP's structure
    print_module_summary(mlp, input_size)

    # Forward pass through the network
    my_input = jax.random.normal(input_rng, input_size)
    my_output = mlp.apply(params, my_input)
    assert my_output.shape[-1] == layer_sizes[-1]

    # Check number of parameters
    num_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    expected_num_params = 0
    sizes = input_size + layer_sizes
    for i in range(len(sizes) - 1):
        expected_num_params += sizes[i] * sizes[i + 1]  # weights
        expected_num_params += sizes[i + 1]  # biases
    assert num_params == expected_num_params


def test_mlp_save_load() -> None:
    """Verify that we can pickle an MLP."""
    rng = jax.random.PRNGKey(0)
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


if __name__ == "__main__":
    test_mlp_construction()
    test_mlp_save_load()
    test_action_sequence_mlp()
