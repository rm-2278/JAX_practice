import jax
import jax.numpy as jnp

class Counter:
    state: int
    
    def __init__(self):
        self.state = 0
    
    def count(self):
        self.state += 1
        return self.state
    
    
x = Counter()
countjit = jax.jit(x.count)
print(countjit())
print(countjit())
print(countjit())


class StatelessCounter:
    
    def count(self, state):
        state += 1
        return state
    
x = StatelessCounter()
state = 0
countjit = jax.jit(x.count)
for i in range(3):
    state = countjit(state)
    print(state)