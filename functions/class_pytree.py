import jax
import jax.numpy as jnp
from jax import jit
from jax import tree_util

class MyClass():
    def __init__(self, x: jnp.ndarray, mul:bool):
        self.x = x
        self.mul = mul
    
    @jit
    def calc(self, y):
        if self.mul:
            return self.x * y
        return y
    
    def _tree_flatten(self):
        children = (self.x, ) #mutable
        aux_data = {'mul': self.mul} #immutable (checked for change when caching)
        return (children, aux_data)
        
    def _tree_unflatten(cls, children, aux_data):
        return cls(*children, **aux_data)
        
        
    
tree_util.register_pytree_node(MyClass, MyClass._tree_flatten, MyClass._tree_unflatten)
        