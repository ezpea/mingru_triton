# mingru_triton

This package contains a Triton-based, chunked implementation of a
simple multiplicative recurrent unit (MinGRU) and a small test harness
I made while trying to learn triton (using Gemini/GPT).

What is included
- `mingru_triton.py` - Triton kernels + a combined PyTorch autograd
	Function (entry point `mingru_chunked`).
- `test/` - pytest-based tests and reference implementations used for
	validation and debugging.

Running tests
1. Ensure you have a CUDA-capable GPU, PyTorch and Triton installed. There is no CPU fallback.
2. From the package root run (this project expects to be runnable with
	 the package on PYTHONPATH):

```bash
PYTHONPATH=`pwd` pytest -q -s test/test_triton_matches_reference.py
```

Notes
- The package exposes `mingru_chunked` at import-time; import it as
	`from mingru_triton import mingru_chunked`.
- Tests may be skipped when CUDA is not available.

If you need other exports for debugging (intermediates, kernels), you
can import the module directly. 

### Usage example
-------------
Quick snippet showing how to call the high-level API and compare the two
backward aggregation modes (host sequential vs. Triton associative scan):

```python
import torch
from mingru_triton import mingru_chunked

device = torch.device('cuda')
B, T, D = 1, 8, 4
T_chunk_size = 2
D_chunk_size = 4

a_init = 0.5 * torch.ones((B, T, D), device=device, dtype=torch.float32)
b_init = torch.ones((B, T, D), device=device, dtype=torch.float32)

a = a_init.clone().requires_grad_()
b = b_init.clone().requires_grad_()
h_tri = mingru_chunked(a, b, T_chunk_size, D_chunk_size, backward_use_scan=use_scan)
(h_tri.sum()).backward()
```

### No CPU fallback
mingru_triton does not provide a useful CPU based fallback. You can try something like
```python
def mingru(a, b, T_chunk_size, D_chunk_size, backward_use_scan=False):
    """
    High-level dispatch: use Triton chunked implementation for CUDA tensors,
    fall back to reference PyTorch implementation on CPU.
    """
    if not a.is_cuda or not torch.cuda.is_available():
        # import the reference forward implementation (pure PyTorch)
        from test.mingru_reference import mingru_reference_forward
        a_ref = a.detach().clone().requires_grad_(a.requires_grad)
        b_ref = b.detach().clone().requires_grad_(b.requires_grad)
        return mingru_reference_forward(a_ref, b_ref)[2]  # returns (a,b,h), return h
    # CUDA path uses existing Triton API
    return mingru_chunked(a, b, T_chunk_size, D_chunk_size, backward_use_scan)
```
