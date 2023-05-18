# Neural ODEs
This is a simple reimplementation of [Neural Ordinary Differential Equations](https://arxiv.org/abs/1806.07366) (Neural ODEs), a fascinating class of neural networks that utilize the power of differential equations to update their hidden layers. This project is primarily for learning purposes and is based on the work of [1] and [2]. The main goal is to investigate the memory and time behaviors of neural ODEs (see [1, Section 3]) and compare the adjoint sensitivity algorithm and direct backpropagation through ODE solvers.


# Installation
To install requirements:
```bash
pip install -r requirements.txt
```

# References
[1] Ricky T. Q. Chen, Yulia Rubanova, Jesse Bettencourt, David Duvenaud. "Neural Ordinary Differential Equations." Advances in Neural Processing Information Systems. 2018.

[2] Ricky T. Q. Chen. torchdiffeq [Computer software]. 2018. https://github.com/rtqichen/torchdiffeq
