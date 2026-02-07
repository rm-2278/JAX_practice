import jax
from jax.sharding import PartitionSpec as P

jax.config.update('jax_num_cpu_devices', 8)
print(jax.devices())
arr = jax.numpy.zeros((4, 8))
print(arr.sharding)
print(jax.debug.visualize_array_sharding(arr))

# mesh = jax.make_mesh((2, 4), ('x', 'y'))
# sharding = jax.sharding.NamedSharding(mesh, P(('x', 'y')))
# print(sharding)

# arr_sharded = jax.device_put(arSr, sharding)
# print(jax.debug.visualize_array_sharding(arr_sharded))

print(jax.typeof(arr))