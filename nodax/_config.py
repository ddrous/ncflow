import jax

# jax.config.update("jax_debug_nans", True)
# jax.config.update("jax_platform_name", "cpu")

print("\n############# Inductive Bias Learning for Dynamical Systems #############\n")
print("Jax version:", jax.__version__)
print("Available devices:", jax.devices())

