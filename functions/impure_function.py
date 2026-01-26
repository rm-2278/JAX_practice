import jax

@jax.jit
def impure_function(x):
    print("This is a side effect which is not functional")
    return x

print("First call: ", impure_function(4.))
print("Second call: ", impure_function(5.))

print("Type or shape change", impure_function(jax.numpy.array([5.])))
print("Type or shape change", impure_function(jax.numpy.array([5])))

@jax.jit
def impure_function_with_global(x):
    print("Global")
    return x + u

u = 5.
print("First call: ", impure_function_with_global(4.))
u = 10.
print("Second call: ", impure_function_with_global(5.))
print("Type or shape change", impure_function_with_global(jax.numpy.array([5.])))

g = 0.
@jax.jit
def impure_saves_normal(x):
    global g
    g = x
    return x

print("First call: ", impure_saves_normal(4.))
print(g) #Tracer value from when tracing
print("Second call: ", impure_saves_normal(5.))
print(g) #No tracing so value unchanged



from jax import lax

array = jax.numpy.arange(10)
print(lax.fori_loop(0, 10, lambda i, x: x + array[i], 0))
iterator = iter(range(10))
print(lax.fori_loop(0, 10, lambda i, x: x + next(iterator), 0)) # 0 because only traced once.


