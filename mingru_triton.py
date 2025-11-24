import torch
import triton
import triton.language as tl
import math

# --- Shared Kernel: Associative Scan Combine ---
@triton.jit
def _scan_combine(a_prev, b_prev, a_curr, b_curr):
    # Combine rule:
    # New Multiplier = Curr_A * Prev_A
    # New Additive   = Curr_A * Prev_B + Curr_B
    acc_a = a_curr * a_prev
    acc_b = a_curr * b_prev + b_curr
    return acc_a, acc_b


# --- FORWARD Phase 1: Local Scan & Summary ---
@triton.jit
def _mingru_scan_phase1_kernel(
    a_ptr, b_ptr, local_h_ptr, block_a_ptr, block_b_ptr,
    n_seq, n_dim, T_chunk_size,
    stride_a_b, stride_a_t, stride_a_d,
    stride_b_b, stride_b_t, stride_b_d,
    stride_lh_b, stride_lh_t, stride_lh_d,
    stride_ba_b, stride_ba_t, stride_ba_d,
    stride_bb_b, stride_bb_t, stride_bb_d,
    D_chunk_size: tl.constexpr
):
    pid_batch = tl.program_id(0)
    pid_chunk = tl.program_id(1)
    pid_dim = tl.program_id(2)
    offs_dim = pid_dim * D_chunk_size + tl.arange(0, D_chunk_size)
    mask_dim = offs_dim < n_dim
    t_start = pid_chunk * T_chunk_size
    
    a_base = a_ptr + pid_batch * stride_a_b + offs_dim * stride_a_d
    b_base = b_ptr + pid_batch * stride_b_b + offs_dim * stride_b_d
    lh_base = local_h_ptr + pid_batch * stride_lh_b + offs_dim * stride_lh_d

    curr_h = tl.zeros([D_chunk_size], dtype=tl.float32)    
    acc_a = tl.full([D_chunk_size], 1.0, dtype=tl.float32)
                
    # Now run the local scan
    for k in range(T_chunk_size):
        t = t_start + k
        if t < n_seq:            
            a_val = tl.load(a_base + t * stride_a_t, mask=mask_dim, other=1.0).to(tl.float32)
            b_val = tl.load(b_base + t * stride_b_t, mask=mask_dim, other=0.0).to(tl.float32)
            
            # Recurrence: h_t = a_t * h_{t-1} + b_t
            curr_h = a_val * curr_h + b_val

            # Calculate Block A (Multiplicative summary A_{t_{start}..t_{end}})
            acc_a = a_val * acc_a
            
            # Store local hidden state 
            tl.store(lh_base + t * stride_lh_t, curr_h, mask=mask_dim)
            
    # Store Forward Summaries
    ba_ptr_loc = block_a_ptr + pid_batch * stride_ba_b + pid_chunk * stride_ba_t + offs_dim * stride_ba_d
    bb_ptr_loc = block_b_ptr + pid_batch * stride_bb_b + pid_chunk * stride_bb_t + offs_dim * stride_bb_d
    tl.store(ba_ptr_loc, acc_a, mask=mask_dim)
    tl.store(bb_ptr_loc, curr_h, mask=mask_dim)


# --- FORWARD/BACKWARD Phase 2: Inter-Chunk Scan (Prefix Sum) ---
@triton.jit
def _mingru_scan_phase2_kernel(
    block_a_ptr, block_b_ptr, chunk_h0_ptr,
    n_chunks, n_dim,
    stride_ba_b, stride_ba_c, stride_ba_d,
    stride_bb_b, stride_bb_c, stride_bb_d,
    stride_ch0_b, stride_ch0_c, stride_ch0_d,
    T_num_chunks_pow_2: tl.constexpr,
    D_chunk_size: tl.constexpr
):
    pid_batch = tl.program_id(0)
    pid_dim = tl.program_id(1)

    # Dimension blocking logic
    offs_dim = pid_dim * D_chunk_size + tl.arange(0, D_chunk_size)
    mask_dim = offs_dim < n_dim
    
    offs_chunks = tl.arange(0, T_num_chunks_pow_2)
    mask_chunks = offs_chunks < n_chunks
    
    # Load A and B components (tensor of size [T_num_chunks_pow_2, D_chunk_size])
    a_ptr = block_a_ptr + pid_batch * stride_ba_b + offs_chunks[:, None] * stride_ba_c + offs_dim[None, :] * stride_ba_d
    b_ptr = block_b_ptr + pid_batch * stride_bb_b + offs_chunks[:, None] * stride_bb_c + offs_dim[None, :] * stride_bb_d
    
    a_vals = tl.load(a_ptr, mask=mask_chunks[:, None] & mask_dim[None, :], other=1.0).to(tl.float32)
    b_vals = tl.load(b_ptr, mask=mask_chunks[:, None] & mask_dim[None, :], other=0.0).to(tl.float32)
    
    # Associative Scan (Prefix Sum) over axis 0 (chunks)
    acc_a, acc_b = tl.associative_scan((a_vals, b_vals), 0, _scan_combine)
    
    # Store the results to chunk_h0[c] = h_{c-1} or G_{c}^{h0}
    # Note: acc_b[k] holds P*[k]. We store P*[k] at Temp[k].
    dest_indices = offs_chunks
    dest_mask = offs_chunks < n_chunks
    
    ch0_base = chunk_h0_ptr + pid_batch * stride_ch0_b + offs_dim[None, :] * stride_ch0_d
    dest_ptr = ch0_base + dest_indices[:, None] * stride_ch0_c
    
    # Store the prefix-sum B component result 
    # Use 2D mask combining chunk mask and dimension mask
    tl.store(dest_ptr, acc_b, mask=dest_mask[:, None] & mask_dim[None, :])


# --- FORWARD Phase 3: Distribute & Finalize ---
@triton.jit
def _mingru_scan_phase3_kernel(
    local_h_ptr, chunk_h0_ptr, a_ptr,
    n_seq, n_dim, T_chunk_size,
    stride_lh_b, stride_lh_t, stride_lh_d,
    stride_ch0_b, stride_ch0_t, stride_ch0_d,
    stride_a_b, stride_a_t, stride_a_d,
    D_chunk_size: tl.constexpr
):
    pid_batch = tl.program_id(0)
    pid_chunk = tl.program_id(1)
    pid_dim = tl.program_id(2)
    
    offs_dim = pid_dim * D_chunk_size + tl.arange(0, D_chunk_size)
    mask_dim = offs_dim < n_dim
    
    t_start = pid_chunk * T_chunk_size

    # Load the global initial state h0 for this chunk
    ch0_base = chunk_h0_ptr + pid_batch * stride_ch0_b + pid_chunk * stride_ch0_t + offs_dim * stride_ch0_d
    h0_val = tl.load(ch0_base, mask=mask_dim, other=0.0).to(tl.float32)
    
    running_a_prod = tl.full([D_chunk_size], 1.0, dtype=tl.float32)
    
    a_base = a_ptr + pid_batch * stride_a_b + offs_dim * stride_a_d
    lh_base = local_h_ptr + pid_batch * stride_lh_b + offs_dim * stride_lh_d

    for k in range(T_chunk_size):
        t = t_start + k
        if t < n_seq:
            a_val = tl.load(a_base + t * stride_a_t, mask=mask_dim, other=1.0).to(tl.float32)
            
            # Update the accumulated product: A_{t_{start}..t} = A_{t_{start}..t-1} * a_t
            running_a_prod = running_a_prod * a_val
            
            loc_h_ptr = lh_base + t * stride_lh_t
            val_h = tl.load(loc_h_ptr, mask=mask_dim).to(tl.float32)
            
            # Final H_t = Local H_t + (A_{t_{start}..t} * H0)
            final_h = val_h + (running_a_prod * h0_val)
            
            tl.store(loc_h_ptr, final_h, mask=mask_dim)


# --- BACKWARD Phase 1: Local Grad Scan & Summary ---
@triton.jit
def _mingru_backward_phase1_kernel(
    a_ptr, grad_out_ptr, 
    block_a_rev_ptr, block_b_rev_ptr,
    n_seq, n_dim, T_chunk_size,
    stride_a_b, stride_a_t, stride_a_d,
    stride_go_b, stride_go_t, stride_go_d,
    stride_ba_b, stride_ba_t, stride_ba_d,
    stride_bb_b, stride_bb_t, stride_bb_d,
    D_chunk_size: tl.constexpr
):
    pid_batch = tl.program_id(0)
    pid_chunk = tl.program_id(1)
    pid_dim = tl.program_id(2)

    offs_dim = pid_dim * D_chunk_size + tl.arange(0, D_chunk_size)
    mask_dim = offs_dim < n_dim
    
    t_end_incl = tl.minimum((pid_chunk + 1) * T_chunk_size, n_seq) - 1
    t_start = pid_chunk * T_chunk_size
    
    a_base = a_ptr + pid_batch * stride_a_b + offs_dim * stride_a_d
    go_base = grad_out_ptr + pid_batch * stride_go_b + offs_dim * stride_go_d

    grad_h_accum = tl.zeros([D_chunk_size], dtype=tl.float32) # G_{t+1}^{\text{local}}
    acc_a_rev = tl.full([D_chunk_size], 1.0, dtype=tl.float32) # A_{t..t_{end}}

    # Loop backward in time
    for k in range(T_chunk_size):
        t = t_end_incl - k
        
        if t >= t_start and t < n_seq:
            grad_out_val = tl.load(go_base + t * stride_go_t, mask=mask_dim, other=0.0).to(tl.float32)
            
            # a_t is needed for the reverse product (acc_a_rev)
            a_curr_val = tl.load(a_base + t * stride_a_t, mask=mask_dim, other=1.0).to(tl.float32)
            
            # a_{t+1} is needed for the gradient recurrence (G_{t+1} -> G_t)
            a_next_val = tl.full([D_chunk_size], 1.0, dtype=tl.float32)
            if t < n_seq - 1:
                a_next_val = tl.load(a_base + (t + 1) * stride_a_t, mask=mask_dim, other=1.0).to(tl.float32)
            
            # Recurrence: G_{t}^{\text{local}} = grad_out[t] + a_{t+1} G_{t+1}^{\text{local}}
            grad_h_accum = a_next_val * grad_h_accum + grad_out_val
            
            # Update the reverse cumulative product (A_{t..t_{end}})
            acc_a_rev = a_curr_val * acc_a_rev
            
    # Store Backward Summaries (in forward chunk order)
    ba_ptr_loc = block_a_rev_ptr + pid_batch * stride_ba_b + pid_chunk * stride_ba_t + offs_dim * stride_ba_d
    bb_ptr_loc = block_b_rev_ptr + pid_batch * stride_bb_b + pid_chunk * stride_bb_t + offs_dim * stride_bb_d
    
    # acc_a_rev is A_{t_{start}..t_{end}}
    tl.store(ba_ptr_loc, acc_a_rev, mask=mask_dim)
    
    # grad_h_accum is G_{t_{start}}^{\text{local}}
    tl.store(bb_ptr_loc, grad_h_accum, mask=mask_dim)


# --- BACKWARD Phase 3: Distribute Global Grad & Calc Final Gradients ---
@triton.jit
def _mingru_backward_phase3_kernel(
    h_final_ptr, a_ptr, chunk_grad_h0_ptr, grad_out_ptr, 
    grad_a_ptr, grad_b_ptr,
    n_seq, n_dim, T_chunk_size,
    stride_hf_b, stride_hf_t, stride_hf_d,
    stride_a_b, stride_a_t, stride_a_d,
    stride_ch0_b, stride_ch0_t, stride_ch0_d,
    stride_go_b, stride_go_t, stride_go_d,
    stride_ga_b, stride_ga_t, stride_ga_d,
    stride_gb_b, stride_gb_t, stride_gb_d,
    D_chunk_size: tl.constexpr
):
    pid_batch = tl.program_id(0)
    pid_chunk = tl.program_id(1)
    pid_dim = tl.program_id(2)
    offs_dim = pid_dim * D_chunk_size + tl.arange(0, D_chunk_size)
    mask_dim = offs_dim < n_dim
    
    t_start = pid_chunk * T_chunk_size
    t_end_incl = tl.minimum((pid_chunk + 1) * T_chunk_size, n_seq) - 1
    #triton.device_print("pid_batch", pid_batch, "t_start", t_end_incl)

    # Load the global gradient flowing into the end of this chunk (G_{t_{end}+1}^{\text{total}})
    ch0_base = chunk_grad_h0_ptr + pid_batch * stride_ch0_b + pid_chunk * stride_ch0_t + offs_dim * stride_ch0_d
    running_grad_h_total = tl.load(ch0_base, mask=mask_dim, other=0.0).to(tl.float32)

    # Base pointers
    hf_base = h_final_ptr + pid_batch * stride_hf_b + offs_dim * stride_hf_d
    a_base = a_ptr + pid_batch * stride_a_b + offs_dim * stride_a_d
    go_base = grad_out_ptr + pid_batch * stride_go_b + offs_dim * stride_go_d 
    ga_base = grad_a_ptr + pid_batch * stride_ga_b + offs_dim * stride_ga_d
    gb_base = grad_b_ptr + pid_batch * stride_gb_b + offs_dim * stride_gb_d

    # Iterate backward over the chunk
    for k in range(T_chunk_size):
        t = t_end_incl - k
        
        if t >= t_start and t < n_seq:
            grad_out_val = tl.load(go_base + t * stride_go_t, mask=mask_dim, other=0.0).to(tl.float32) 
            a_curr_val = tl.load(a_base + t * stride_a_t, mask=mask_dim, other=1.0).to(tl.float32) # a_t

            # Load h_{t-1} for grad_a[t] calculation
            h_prev = tl.full([D_chunk_size], 0.0, dtype=tl.float32)
            if t > 0:
                h_prev = tl.load(hf_base + (t-1) * stride_hf_t, mask=mask_dim, other=0.0).to(tl.float32)

            # 1. Determine a_{t+1} (Multiplier for G_{t+1} -> G_t)
            a_next_val = tl.full([D_chunk_size], 1.0, dtype=tl.float32)
            if t < n_seq - 1:
                a_next_val = tl.load(a_base + (t + 1) * stride_a_t, mask=mask_dim, other=1.0).to(tl.float32)
            
            # 2. Calculate the total gradient G_t^{\text{total}}:
            # G_t^{\text{total}} = grad\_out[t] + a_{t+1} G_{t+1}^{\text{total}}
            grad_h_total = grad_out_val + a_next_val * running_grad_h_total

            # 3. Calculate final gradients dL/da_t and dL/db_t
            # grad_b[t] = G_t^{\text{total}}
            tl.store(gb_base + t * stride_gb_t, grad_h_total, mask=mask_dim)

            # grad_a[t] = G_t^{\text{total}} * h_{t-1}
            tl.store(ga_base + t * stride_ga_t, grad_h_total * h_prev, mask=mask_dim)
            
            # 4. Update the running total for the next (earlier) time step
            # The next iteration needs G_t^{total} (not a_t * G_t^{total})
            # because the recurrence is G_{t-1}^{total} = grad_out[t-1] + a_t * G_t^{total}.
            running_grad_h_total = grad_h_total
            
# --- Autograd Function (Combined) ---
class MinGRUChunkedFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, T_chunk_size, D_chunk_size, backward_use_scan: bool = False):
        a = a.contiguous()
        b = b.contiguous()
        B, T, D = a.shape
        T_num_chunks = math.ceil(T / T_chunk_size)
        
        # Intermediate Tensors
        local_h = torch.empty_like(b)
        block_a = torch.empty((B, T_num_chunks, D), device=a.device, dtype=torch.float32)
        block_b = torch.empty((B, T_num_chunks, D), device=a.device, dtype=torch.float32)
        chunk_h0 = torch.zeros((B, T_num_chunks, D), device=a.device, dtype=torch.float32)
        
        grid_p1 = (B, T_num_chunks, triton.cdiv(D, D_chunk_size))
        
        # Phase 1: Local Scan & Block Summaries
        _mingru_scan_phase1_kernel[grid_p1](
            a, b, 
            local_h, block_a, block_b, 
            T, D, T_chunk_size, 
            a.stride(0), a.stride(1), a.stride(2), 
            b.stride(0), b.stride(1), b.stride(2), 
            local_h.stride(0), local_h.stride(1), local_h.stride(2), 
            block_a.stride(0), block_a.stride(1), block_a.stride(2), 
            block_b.stride(0), block_b.stride(1), block_b.stride(2), 
            D_chunk_size=D_chunk_size)
        
        # Phase 2: Inter-Chunk Scan (Prefix Sum)
        grid_p2 = (B, triton.cdiv(D, D_chunk_size)) 
        T_num_chunks_pow_2 = triton.next_power_of_2(T_num_chunks)
        
        # Note on chunk_h0: This tensor receives the result of the prefix sum, P*[k], 
        # stored at index k. This is used in Phase 3 to get the global state H0.
        _mingru_scan_phase2_kernel[grid_p2](
            block_a, block_b, 
            chunk_h0, T_num_chunks, D, 
            block_a.stride(0), block_a.stride(1), block_a.stride(2), 
            block_b.stride(0), block_b.stride(1), block_b.stride(2), 
            chunk_h0.stride(0), chunk_h0.stride(1), chunk_h0.stride(2), 
            T_num_chunks_pow_2=T_num_chunks_pow_2,
            D_chunk_size=D_chunk_size
        )
        
        # Phase 3: Distribute Global State
        # The true H0 for chunk c is chunk_h0[c-1]. We correct this by shifting 
        # the indices later in this function, but for now we run the base kernel 
        # which uses a slightly simpler recurrence
        # H_t = Local H_t + (A_{t_{start}..t} * H0)
        
        # We need to shift the results in chunk_h0 so that chunk_h0[c] = P*[c-1] (the required H0)
        # The kernel stored P*[k] at index k.
        # Required: H0[c] = P*[c-1]
        
        # P*[k] is stored at chunk_h0[k]. We need chunk_h0[c-1] for chunk c.
        # P*[0] = H_0. P*[c] = H_c (end of chunk c).
        
        # H0[c] must be H_{c-1} = P*[c-1].
        # H0[0] is 0 (already set).
        
        # Shift 1: Move P*[c] to H0[c+1].
        # H0[c] = chunk_h0[c-1] for c=1 to N-1
        
        # We use a temporary copy to handle the index shift cleanly on the host side
        H0_shifted = torch.zeros_like(chunk_h0)
        if T_num_chunks > 1:
            H0_shifted[:, 1:, :] = chunk_h0[:, :-1, :]
            
        _mingru_scan_phase3_kernel[grid_p1](
            local_h, H0_shifted, a, 
            T, D, T_chunk_size, 
            local_h.stride(0), local_h.stride(1), local_h.stride(2), 
            H0_shifted.stride(0), H0_shifted.stride(1), H0_shifted.stride(2), 
            a.stride(0), a.stride(1), a.stride(2), 
            D_chunk_size=D_chunk_size)
        
        h_final = local_h
        ctx.save_for_backward(a, b, h_final) 
        ctx.T_chunk_size = T_chunk_size
        ctx.D_chunk_size = D_chunk_size
        ctx.T = T
        ctx.D = D
        ctx.T_num_chunks = T_num_chunks
        ctx.backward_use_scan = backward_use_scan
        return h_final

    @staticmethod
    def backward(ctx, grad_out):
        a, b, h_final = ctx.saved_tensors
        T_chunk_size, D_chunk_size, T, D, T_num_chunks = ctx.T_chunk_size, ctx.D_chunk_size, ctx.T, ctx.D, ctx.T_num_chunks
        grad_out = grad_out.contiguous()

        grad_a = torch.empty_like(a)
        grad_b = torch.empty_like(a)
        
        # Tensors for the backward recurrence (A_rev, B_rev)
        block_a_rev = torch.empty((a.shape[0], T_num_chunks, D), device=a.device, dtype=torch.float32)
        block_b_rev = torch.empty((a.shape[0], T_num_chunks, D), device=a.device, dtype=torch.float32)
        
        # Stores G_{t_{end}+1}^{\text{total}} for each chunk
        chunk_grad_h0 = torch.zeros((a.shape[0], T_num_chunks, D), device=a.device, dtype=torch.float32) 
        
        # Temporary buffer for the prefix scan result
        # Note: This kernel stores P*[k] at index k.
        chunk_grad_h0_temp = torch.zeros_like(chunk_grad_h0)

        grid_p1 = (a.shape[0], T_num_chunks, triton.cdiv(D, D_chunk_size))
        
        # --- 1. Backward Phase 1: Local Grad Scan & Summary ---
        _mingru_backward_phase1_kernel[grid_p1](
            a, grad_out, block_a_rev, block_b_rev,
            T, D, T_chunk_size,
            a.stride(0), a.stride(1), a.stride(2),
            grad_out.stride(0), grad_out.stride(1), grad_out.stride(2),
            block_a_rev.stride(0), block_a_rev.stride(1), block_a_rev.stride(2),
            block_b_rev.stride(0), block_b_rev.stride(1), block_b_rev.stride(2),
            D_chunk_size=D_chunk_size
        )
        
        # --- 2. Compute chunk-wise global gradients (host-side loop) ---
        # The Triton prefix-scan approach is subtle to map correctly for the
        # backward (suffix) case across arbitrary chunk counts. To ensure
        # correctness across all (T, T_chunk_size) combinations we compute the
        # chunk-level G_{t_start}^{total} on the host using the per-chunk
        # reverse summaries produced by Phase1 (block_a_rev, block_b_rev).
        # This is implemented as a vectorized loop over chunks on CUDA and
        # matches the reference implementation.
        if T_num_chunks > 1:
            if getattr(ctx, "backward_use_scan", False):
                # Use Triton associative scan on flipped reverse-block summaries.
                # We flip the chunk axis, run the same scan kernel used in
                # forward, then flip results back to original order.
                T_num_chunks_pow_2 = triton.next_power_of_2(T_num_chunks)
                grid_p2 = (a.shape[0], triton.cdiv(D, D_chunk_size))

                block_a_rev_flipped = torch.flip(block_a_rev, [1])
                block_b_rev_flipped = torch.flip(block_b_rev, [1])

                # chunk_grad_h0_temp will receive P*[k] in flipped order
                _mingru_scan_phase2_kernel[grid_p2](
                    block_a_rev_flipped, block_b_rev_flipped, chunk_grad_h0_temp,
                    T_num_chunks, D,
                    block_a_rev_flipped.stride(0), block_a_rev_flipped.stride(1), block_a_rev_flipped.stride(2),
                    block_b_rev_flipped.stride(0), block_b_rev_flipped.stride(1), block_b_rev_flipped.stride(2),
                    chunk_grad_h0_temp.stride(0), chunk_grad_h0_temp.stride(1), chunk_grad_h0_temp.stride(2),
                    T_num_chunks_pow_2=T_num_chunks_pow_2,
                    D_chunk_size=D_chunk_size
                )

                # Flip back to forward chunk order so chunk_grad_h0[c] = G_total at t_start_of_chunk_c
                chunk_grad_h0 = torch.flip(chunk_grad_h0_temp, [1])
            else:
                # Compute the full total-gradient sequence G_t (device-side host loop)
                B0 = a.shape[0]
                G_total = torch.zeros_like(h_final, dtype=torch.float32)
                G_next = torch.zeros((B0, D), device=a.device, dtype=torch.float32)
                for t in range(T - 1, -1, -1):
                    a_next = torch.ones((B0, D), device=a.device, dtype=torch.float32)
                    if t < T - 1:
                        a_next = a[:, t + 1, :]
                    grad_out_val = grad_out[:, t, :]
                    G_t = grad_out_val + a_next * G_next
                    G_total[:, t, :] = G_t
                    G_next = G_t

                # Map: chunk_grad_h0[c] = G_total[:, t_start_of_chunk_c] for c>0
                for c in range(1, T_num_chunks):
                    t_start = c * T_chunk_size
                    if t_start < T:
                        chunk_grad_h0[:, c, :] = G_total[:, t_start, :]
            
        # --- 3. Backward Phase 3: Distribute Global Grad & Calc Final Gradients ---
        # Shift chunk_grad_h0 so that chunk_grad_h0_for_phase3[c] contains the
        # gradient flowing into the end of chunk c (i.e., G_{t_end+1}), which
        # corresponds to G_ref at the start of the next chunk. The last chunk
        # naturally receives zero.
        chunk_grad_h0_for_phase3 = torch.zeros_like(chunk_grad_h0)
        if T_num_chunks > 1:
            chunk_grad_h0_for_phase3[:, :-1, :] = chunk_grad_h0[:, 1:, :]

        _mingru_backward_phase3_kernel[grid_p1](
            h_final, a, chunk_grad_h0_for_phase3, grad_out, 
            grad_a, grad_b,
            T, D, T_chunk_size,
            h_final.stride(0), h_final.stride(1), h_final.stride(2),
            a.stride(0), a.stride(1), a.stride(2),
            chunk_grad_h0.stride(0), chunk_grad_h0.stride(1), chunk_grad_h0.stride(2),
            grad_out.stride(0), grad_out.stride(1), grad_out.stride(2),
            grad_a.stride(0), grad_a.stride(1), grad_a.stride(2),
            grad_b.stride(0), grad_b.stride(1), grad_b.stride(2),
            D_chunk_size=D_chunk_size
        )
        
        # Store intermediate tensors
        ctx.triton_backward_intermediates = (block_a_rev, block_b_rev, chunk_grad_h0, chunk_grad_h0_for_phase3)
        
        return grad_a, grad_b, None, None, None

def mingru_chunked(a, b, T_chunk_size, D_chunk_size, backward_use_scan=True):
    return MinGRUChunkedFunction.apply(a, b, T_chunk_size, D_chunk_size, backward_use_scan)