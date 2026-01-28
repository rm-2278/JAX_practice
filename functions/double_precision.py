import jax
import jax.numpy as jnp
import sys
# jax.config.update("jax_enable_x64", True)
# jax.config.config_with_absl()

def main():
    x = jax.random.uniform(jax.random.key(0), (10, ), dtype="float64")
    print(x.dtype)


if __name__ == "__main__":
    jax.config.parse_flags_with_absl() # Implicitly passed args
    main()