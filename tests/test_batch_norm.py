from flax import nnx
import jax.numpy as jnp
import jax

if __name__=="__main__":
    rng = jax.random.key(0)

    # Make some fake data
    rng, gen_rng = jax.random.split(rng)
    var = jnp.array([0.1, 1.0, 5.9])
    y = var * jax.random.normal(gen_rng, (1024, 3))

    # Check the emperical mean and variance
    mean = jnp.mean(y, axis=0)
    var = jnp.var(y, axis=0)
    print("mean", mean)
    print("var", var)

    # Create a batch norm layer
    normalizer = nnx.BatchNorm(
        num_features=3,
        use_bias=False,
        use_scale=False,
        use_fast_variance=False,
        rngs=nnx.Rngs(0),
    )

    # Normalize the data
    y_norm = normalizer(y)
    print("normalized mean", jnp.mean(y_norm, axis=0))
    print("normalized var", jnp.var(y_norm, axis=0))

    print("normalizer mean", normalizer.mean.value)
    print("normalizer var", normalizer.var.value)