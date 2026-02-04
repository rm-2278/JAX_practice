# JAX_practice

This repository contains basic to modern machine learning models using JAX, mostly from scratch.
It also contains code that is used to consolidate the understanding of each function from JAX.



## AutoEncoder
Created an AutoEncoder with JAX.
- Compared how reconstruction loss changes as latent dim changes.
- Using asynchronous dispatch & avoiding sync within loop (together with the built-in XLA feature of JAX), maximised usage of GPU (~95%).