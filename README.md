# Adam from Scratch

A reimplementation of the Adam optimizer using PyTorch’s low-level API. In this repo, I reproduced the key behavior of `torch.optim.Adam`, including:

- bias-corrected first and second moment estimates
- support for arbitrary parameter shapes
- weight decay (L2 regularization)
- numerical stability via `eps`
- compatibility with PyTorch’s `.step()` and `.zero_grad()` training loops

This project was built to deeply understand how Adam works under the hood  
(independent of PyTorch's built-in optimizer).

See `notebooks/adam_demo.ipynb` for a full training demo (training a simple linear regression and 2 layer XOR MLP) and comparison with PyTorch’s Adam.

---

### Tests

All tests pass via `pytest` and the test suite checks for:

- output equivalence to `torch.optim.Adam` (via `torch.allclose` with very strict absolute tolerance)
- Correct handling of `.grad == None` and multiple parameter tensors
- loss convergence on linear regression tasks
