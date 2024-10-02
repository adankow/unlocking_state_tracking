# -*- coding: utf-8 -*-

import pytest
import torch
import torch.nn.functional as F

from fla.ops.gla import chunk_gla, fused_chunk_gla, fused_recurrent_gla
from fla.ops.gla.naive import naive_recurrent_gla

@pytest.mark.parametrize("B", [4])
@pytest.mark.parametrize("H", [4])
@pytest.mark.parametrize("T", [300, 512])
@pytest.mark.parametrize("D", [32, 64, 128])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float])
def test_chunk(
    B: int,
    H: int,
    T: int,
    D: int,
    dtype: torch.dtype
):
    torch.manual_seed(42)
    atol = 1e-2 if dtype == torch.float else 1e-1
    # [B, H, T, d_head]
    q = torch.randn((B, H, T, D), dtype=dtype, device='cuda').requires_grad_()
    k = torch.randn((B, H, T, D), dtype=dtype, device='cuda').requires_grad_()
    v = torch.randn((B, H, T, D), dtype=dtype, device='cuda').requires_grad_()
    g = torch.randn((B, H, T, D), dtype=dtype, device='cuda')
    g = F.logsigmoid(g).clamp_min(-3).requires_grad_(True)
    do = torch.randn_like(v)
    h0 = torch.randn((B, H, D, D), dtype=dtype, device='cuda').requires_grad_()
    ref, ref_ht = fused_recurrent_gla(q, k, v, g, initial_state=h0, output_final_state=True)
    ref.backward(do)
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None
    ref_dg, g.grad = g.grad.clone(), None
    ref_dh0, h0.grad = h0.grad.clone(), None

    # triton implementation
    tri, tri_ht = chunk_gla(q, k, v, g, initial_state=h0, output_final_state=True)
    tri.backward(do)
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None
    tri_dg, g.grad = g.grad.clone(), None
    tri_dh0, h0.grad = h0.grad.clone(), None

    assert ref.allclose(tri, 0, atol), f" o diff: {torch.abs(ref - tri).max()}"
    assert ref_ht.allclose(tri_ht, 0, atol), f" o diff: {torch.abs(ref_ht - tri_ht).max()}"
    assert ref_dq.allclose(tri_dq, 0, atol), f"dq diff: {torch.abs(ref_dq - tri_dq).max()}"
    assert ref_dk.allclose(tri_dk, 0, atol), f"dk diff: {torch.abs(ref_dk - tri_dk).max()}"
    assert ref_dv.allclose(tri_dv, 0, atol), f"dv diff: {torch.abs(ref_dv - tri_dv).max()}"
    assert ref_dg.allclose(tri_dg, 0, atol), f"dg diff: {torch.abs(ref_dg - tri_dg).max()}"
    assert ref_dh0.allclose(tri_dh0, 0, atol), f"dg diff: {torch.abs(ref_dh0 - tri_dh0).max()}"


@pytest.mark.parametrize("use_negative_gates", [True, False])
@pytest.mark.parametrize("B", [4])
@pytest.mark.parametrize("H", [4])
@pytest.mark.parametrize("T", [300, 512])
@pytest.mark.parametrize("D", [32, 64, 128])
@pytest.mark.parametrize("dtype", [torch.float])
def test_fused_recurrent(
    B: int,
    H: int,
    T: int,
    D: int,
    use_negative_gates: bool,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    atol = 1e-3
    q = torch.randn((B, H, T, D), dtype=dtype, device='cuda').requires_grad_()
    k = torch.randn((B, H, T, D), dtype=dtype, device='cuda').requires_grad_()
    v = torch.randn((B, H, T, D), dtype=dtype, device='cuda').requires_grad_()
    g = torch.randn((B, H, T, D), dtype=dtype, device='cuda')
    g = F.logsigmoid(g).clamp_min(-3).requires_grad_(True)
    h0 = torch.randn((B, H, D, D), dtype=dtype, device='cuda').requires_grad_()

    do = torch.randn_like(v)
    ref, ref_ht = naive_recurrent_gla(q, k, v, g, initial_state=h0, output_final_state=True, use_negative_gates=use_negative_gates)
    ref.backward(do)
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None
    ref_dg, g.grad = g.grad.clone(), None
    ref_dh0, h0.grad = h0.grad.clone(), None

    tri, tri_ht = fused_recurrent_gla(q, k, v, g, initial_state=h0, output_final_state=True, use_negative_gates=use_negative_gates)
    tri.backward(do)
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None
    tri_dg, g.grad = g.grad.clone(), None
    tri_dh0, h0.grad = h0.grad.clone(), None

    assert ref.allclose(tri, 0, atol), f" o diff: {torch.abs(ref - tri).max()}"
    assert ref_ht.allclose(tri_ht, 0, atol), f" o diff: {torch.abs(ref_ht - tri_ht).max()}"
    assert ref_dq.allclose(tri_dq, 0, atol), f"dq diff: {torch.abs(ref_dq - tri_dq).max()}"
    assert ref_dk.allclose(tri_dk, 0, atol), f"dk diff: {torch.abs(ref_dk - tri_dk).max()}"
    assert ref_dv.allclose(tri_dv, 0, atol), f"dv diff: {torch.abs(ref_dv - tri_dv).max()}"
    assert ref_dg.allclose(tri_dg, 0, atol), f"dg diff: {torch.abs(ref_dg - tri_dg).max()}"
    assert ref_dh0.allclose(tri_dh0, 0, atol), f"dg diff: {torch.abs(ref_dh0 - tri_dh0).max()}"


@pytest.mark.parametrize("B", [4])
@pytest.mark.parametrize("H", [4])
@pytest.mark.parametrize("T", [300, 512])
@pytest.mark.parametrize("D", [32, 64, 128])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float])
def test_fused_chunk(
    B: int,
    H: int,
    T: int,
    D: int,
    dtype: torch.dtype
):
    torch.manual_seed(42)
    atol = 1e-2 if dtype == torch.float else 1e-1
    # [B, H, T, d_head]
    q = torch.randn((B, H, T, D), dtype=dtype, device='cuda').requires_grad_()
    k = torch.randn((B, H, T, D), dtype=dtype, device='cuda').requires_grad_()
    v = torch.randn((B, H, T, D), dtype=dtype, device='cuda').requires_grad_()
    g = torch.randn((B, H, T, D), dtype=dtype, device='cuda')
    g = F.logsigmoid(g).clamp_min(-3).requires_grad_(True)
    do = torch.randn_like(v)
    ref, _ = fused_recurrent_gla(q, k, v, g)
    ref.backward(do)
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None
    ref_dg, g.grad = g.grad.clone(), None

    # triton implementation
    tri, _ = fused_chunk_gla(q, k, v, g)
    tri.backward(do)
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None
    tri_dg, g.grad = g.grad.clone(), None

    assert ref.allclose(tri, 0, atol), f" o diff: {torch.abs(ref - tri).max()}"
    assert ref_dq.allclose(tri_dq, 0, atol), f"dq diff: {torch.abs(ref_dq - tri_dq).max()}"
    assert ref_dk.allclose(tri_dk, 0, atol), f"dk diff: {torch.abs(ref_dk - tri_dk).max()}"
    assert ref_dv.allclose(tri_dv, 0, atol), f"dv diff: {torch.abs(ref_dv - tri_dv).max()}"
    assert ref_dg.allclose(tri_dg, 0, atol), f"dg diff: {torch.abs(ref_dg - tri_dg).max()}"

if __name__ == '__main__':
    import argparse

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')


    parser = argparse.ArgumentParser(description="Run fused recurrent test with specified parameters.")
    
    parser.add_argument('--D', type=int, default=32, help='Dimensionality D')
    parser.add_argument('--B', type=int, default=4, help='Batch size B')
    parser.add_argument('--H', type=int, default=4, help='Number of heads H')
    parser.add_argument('--T', type=int, default=512, help='Timesteps T')
    parser.add_argument('--dtype', type=str, default='float', help='Data type (e.g., float, double)')
    parser.add_argument('--use_negative_gates', type=str2bool, default=True, help='Whether to use negative gates (True or False)')

    args = parser.parse_args()

    # Convert dtype string to actual torch dtype
    dtype = getattr(torch, args.dtype)

    # Call the test function with arguments from argparse
    test_fused_recurrent(D=args.D, B=args.B, H=args.H, T=args.T, dtype=dtype, use_negative_gates=args.use_negative_gates)

    print('test passed')