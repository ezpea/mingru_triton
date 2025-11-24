import torch
import triton
import math

# --- PyTorch Reference Function for Debugging ---
def mingru_reference_backward(a, grad_out, chunk_size, device):
    B, T, D = a.shape
    num_chunks = math.ceil(T / chunk_size)
    
    # 1. Total Gradient Calculation (Sequential loop - G_ref is G_t^total)
    G_ref = torch.zeros_like(a, dtype=torch.float32)
    G_next = torch.zeros((B, D), device=device, dtype=torch.float32) # G_{t+1}
    
    for t in range(T - 1, -1, -1):
        a_next = a[:, t+1, :] if t < T - 1 else torch.ones((B, D), device=device, dtype=torch.float32)
        
        # G_t = grad_out[t] + a_{t+1} * G_{t+1}
        G_t = grad_out[:, t, :] + a_next * G_next
        G_ref[:, t, :] = G_t
        
        # G_{next} for the next iteration (t-1) is the flow from h_t: a_t * G_t
        G_next = G_t

    # 2. Extract Backward Summaries (Reference)
    block_a_rev_ref = torch.empty((B, num_chunks, D), device=device, dtype=torch.float32)
    block_b_rev_ref = torch.empty((B, num_chunks, D), device=device, dtype=torch.float32)
    chunk_grad_h0_ref = torch.zeros((B, num_chunks, D), device=device, dtype=torch.float32)
    
    for c in range(num_chunks):
        t_start = c * chunk_size
        t_end_incl = min((c + 1) * chunk_size, T) - 1
        
        # Block A Reverse (Product A_{t_{start}..t_{end}})
        a_chunk = a[:, t_start:t_end_incl+1, :]
        if a_chunk.numel() > 0:
            a_prod = torch.prod(a_chunk, dim=1)
            block_a_rev_ref[:, c, :] = a_prod
        else:
            block_a_rev_ref[:, c, :] = 1.0 

        # Block B Reverse (Local Gradient Accumulation G_{t_{start}}^{\text{local}})
        local_grad = torch.zeros((B, D), device=device, dtype=torch.float32) # G_{t+1}^{\text{local}}
        
        for t in range(t_end_incl, t_start - 1, -1):
            a_next = a[:, t+1, :] if t < T - 1 else torch.ones((B, D), device=device, dtype=torch.float32)
            local_grad = grad_out[:, t, :] + a_next * local_grad # G_t^{\text{local}}
            
        block_b_rev_ref[:, c, :] = local_grad
        
        # Chunk Grad H0 (G_{t_{start}}^{\text{total}})
        if c > 0:
            t_after_prev_chunk = t_start
            # G_{t_{start}}^{\text{total}} is the total gradient G_t at t_start
            chunk_grad_h0_ref[:, c, :] = G_ref[:, t_after_prev_chunk, :] 
    
    return block_a_rev_ref, block_b_rev_ref, chunk_grad_h0_ref


def mingru_reference_forward(a, b, h_0=None):
    # 1. PyTorch (Sequential loop - Ground Truth)
    B, T, D = a.shape
    h = torch.zeros_like(b)    
    curr_h = h_0 if h_0 is not None else torch.zeros((B, D), device=a.device, dtype=torch.float32)
    for t in range(T):
        curr_h = a[:, t, :] * curr_h + b[:, t, :]
        h[:, t, :] = curr_h    
    return a, b, h
