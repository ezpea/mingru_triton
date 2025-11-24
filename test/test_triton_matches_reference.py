import pytest
import torch
from test.mingru_reference import mingru_reference_forward, mingru_reference_backward
from mingru_triton import mingru_chunked

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/GPU required for Triton tests")

def check_triton_matches_reference(
    B, T, D,
    T_chunk_size,
    D_chunk_size,
    TOL = 1e-4,
    backward_use_scan=True):
    device = torch.device('cuda')

    print(f"\n--- Verification Results (Triton vs PyTorch) ---")
    print(f"B={B}, T={T}, D={D}, T_chunk_size={T_chunk_size} {D_chunk_size=}")

    # Ensure inputs are differentiable
    a_init = 0.5*torch.ones((B, T, D), device=device, dtype=torch.float32)
    b_init = torch.ones((B, T, D), device=device, dtype=torch.float32)

    def pp(nn, xx): print(nn, xx.reshape(-1).cpu().detach().numpy())

    # 1. PyTorch (Sequential loop - Ground Truth)
    a_ref, b_ref, h_ref = mingru_reference_forward(a_init.clone().requires_grad_(), b_init.clone().requires_grad_())
    loss_ref = h_ref.sum()
    loss_ref.backward()
    grad_a_ref, grad_b_ref = a_ref.grad, b_ref.grad
    grad_out_ref = torch.ones_like(h_ref) # The gradient used is 1.0 everywhere
    # Get reference intermediate tensors
    block_a_rev_ref, block_b_rev_ref, chunk_grad_h0_ref = mingru_reference_backward(a_ref.detach(), grad_out_ref, T_chunk_size, device)

    # 2. Triton (Chunked)
    a = a_init.clone().requires_grad_()
    b = b_init.clone().requires_grad_()

    h_tri = mingru_chunked(a, b, T_chunk_size, D_chunk_size, backward_use_scan = backward_use_scan)    
    loss_tri = h_tri.sum()
    loss_tri.backward()    
    grad_a_tri = a.grad.clone()
    grad_b_tri = b.grad.clone()
    
    # Retrieve Triton intermediate tensors from the backward context
    backward_ctx = h_tri.grad_fn
    
    if hasattr(backward_ctx, 'triton_backward_intermediates'):
        block_a_rev_tri, block_b_rev_tri, chunk_grad_h0_tri, chunk_grad_h0_temp  = backward_ctx.triton_backward_intermediates
    else:
        print("Error: Could not retrieve triton_backward_intermediates from h_tri.grad_fn.")
        raise

    # --- DEBUGGING: Compare Intermediates (A_rev, B_rev, Chunk_Grad_H0) ---
    
    # Calculate Flipped Reference Tensors (Inputs to Phase 2)
    block_a_rev_flipped_ref = torch.flip(block_a_rev_ref, [1])
    block_b_rev_flipped_ref = torch.flip(block_b_rev_ref, [1])
    
    # Calculate Triton Flipped Tensors (Inputs used by Phase 2)
    block_a_rev_tri, block_b_rev_tri, _, _ = backward_ctx.triton_backward_intermediates
    block_a_rev_flipped_tri = torch.flip(block_a_rev_tri, [1])
    block_b_rev_flipped_tri = torch.flip(block_b_rev_tri, [1])

    # Check the flipped inputs match (sanity check)
    barf_diff = (block_a_rev_flipped_tri - block_a_rev_flipped_ref).abs().max().item()
    bbrf_diff = (block_b_rev_flipped_tri - block_b_rev_flipped_ref).abs().max().item()

    pp("a_ref              ", a_ref)
    pp("b_ref              ", b_ref)
    pp("h_ref              ", h_ref)
    # 3. Verification
    pp("chunk_grad_h0_ref  ",   chunk_grad_h0_ref)
    pp("chunk_grad_h0_tri  ",   chunk_grad_h0_tri)
    #pp("chunk_grad_h0_tem  ",  chunk_grad_h0_temp)
    pp("grad_a_ref         ",          grad_a_ref)    
    pp("grad_a_tri         ",          grad_a_tri)
    pp("grad_b_ref         ",          grad_b_ref)
    pp("grad_b_tri         ",          grad_b_tri)
    #print("=========>", grad_a_ref.abs().max())
    h_diff = (h_tri - h_ref).abs().max().item()
    ga_diff = (grad_a_tri - grad_a_ref).abs().max().item()
    gb_diff = (grad_b_tri - grad_b_ref).abs().max().item()
    
    # Intermediate Verification (Check Phase 1 and 2 outputs)
    bar_diff = (block_a_rev_tri - block_a_rev_ref).abs().max().item()
    bbr_diff = (block_b_rev_tri - block_b_rev_ref).abs().max().item()
    cgh0_diff = (chunk_grad_h0_tri - chunk_grad_h0_ref).abs().max().item()
    
    print(f"Forward Pass Max Diff (h_final): {h_diff:.6e} (TOL={TOL:.1e})")    
    print(f"\n--- Intermediate Backward Checks ---")
    print(f"Block A Reverse Max Diff (A_rev): {bar_diff:.6e} (TOL={TOL:.1e})")
    print(f"Block B Reverse Max Diff (G_local): {bbr_diff:.6e} (TOL={TOL:.1e})")
    print(f"Flipped A Input Max Diff (F_A): {barf_diff:.6e} (TOL={TOL:.1e})")
    print(f"Flipped B Input Max Diff (F_B): {bbrf_diff:.6e} (TOL={TOL:.1e})")
    print(f"Chunk Grad H0 Max Diff (G_global): {cgh0_diff:.6e} (TOL={TOL:.1e})")
    
    print(f"\n--- Final Gradient Checks ---")
    print(f"Backward Pass Max Diff (grad_a): {ga_diff:.6e} (TOL={TOL:.1e})")
    print(f"Backward Pass Max Diff (grad_b): {gb_diff:.6e} (TOL={TOL:.1e})")
    
    ok = h_diff < TOL and ga_diff < TOL and gb_diff < TOL

    if ok:
        print("\nMatch: True (Forward and Backward match ground truth)")
    else:
        print("\nMatch: False (Mismatch detected - check implementation logic)")
    return ok
        
@pytest.mark.parametrize("T, T_chunk_size, backward_use_scan", [
    (4,1,True), (4,2,True), (8,2,True), (16,4,True), (16,4,False),
])        
def test_triton_matches_reference_param(T, T_chunk_size, backward_use_scan, B=2, D=3, D_chunk_size=4):
    assert check_triton_matches_reference(B=B, T=T, D=D, T_chunk_size=T_chunk_size, D_chunk_size=D_chunk_size, backward_use_scan=backward_use_scan)