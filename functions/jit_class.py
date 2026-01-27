import jax
import jax.numpy as jnp
from jax import jit
from functools import partial

class MyClass():
    def __init__(self, x: jnp.ndarray, mul: bool):
        self.x = x
        self.mul = mul
        
    @partial(jit, static_argnums=0)
    def calc(self, y):
        if self.mul:
            return self.x * y
        return y
    
    def __hash__(self):
        return hash((self.x, self.mul))
    
    def __eq__(self, other):
        return (isinstance(other, MyClass) and self.x == other.x and self.mul == other.mul)

c = MyClass(10, True)
print(c.calc(4))

c.mul = False
print(c.calc(4))
