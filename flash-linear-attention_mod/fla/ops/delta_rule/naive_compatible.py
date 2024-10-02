# -*- coding: utf-8 -*-

import torch
from einops import rearrange


def delta_rule_recurrence(q, k, v, beta, initial_state= None, output_final_state=False):
    b, h, l, d_k = q.shape
    d_v = v.shape[-1]
    o = torch.zeros_like(v)
    
    if initial_state is None:
        S = torch.zeros(b, h, d_k, d_v).to(v)
    else:
        S = initial_state
        
    q = q * (d_k ** -0.5)

    if beta.ndim < v.ndim:
        beta = beta[..., None]

    for i in range(l):
        _k = k[:, :, i]
        _q = q[:, :, i]
        _v = v[:, :, i]
        beta_i = beta[:, :, i]
        _v = _v - (S * _k[..., None]).sum(-2)
        _v = _v * beta_i
        S = S + _k.unsqueeze(-1) * _v.unsqueeze(-2)
        o[:, :, i] = torch.einsum('bhd,bhdm->bhm', _q, S)
        

    final_state = None
    if output_final_state:
        final_state = S
    
    return o, final_state


def delta_rule_chunkwise(q, k, v, beta, chunk_size=32, initial_state= None, output_final_state=False):
    b, h, l, d_k = q.shape
    d_v = v.shape[-1]
    q = q * (d_k ** -0.5)
    v = v * beta[..., None]
    k_beta = k * beta[..., None]
    
    if chunk_size > l:
        chunk_size = l

    assert l % chunk_size == 0

    # note that diagonal is masked.
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), diagonal=0)
    q, k, v, k_beta = map(lambda x: rearrange(x, 'b h (n c) d -> b h n c d', c=chunk_size), [q, k, v, k_beta])
    attn = -(k_beta @ k.transpose(-1, -2)).masked_fill(mask, 0)

    for i in range(1, chunk_size):
        attn[..., i, :i] = attn[..., i, :i] + (attn[..., i, :, None].clone() * attn[..., :, :i].clone()).sum(-2)

    attn = attn + torch.eye(chunk_size, dtype=torch.float, device=q.device)
    # u
    k_cumsum = attn @ v
    # w
    k_cumdecay = attn @ k_beta

    v = k_cumsum
    if initial_state is None:
        S = k.new_zeros(b, h, d_k, d_v)
    else:
        S = initial_state
        
    o = torch.zeros_like(v)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), diagonal=1)
    for i in range(0, l // chunk_size):
        q_i, k_i, v_i = q[:, :, i], k[:, :, i], v[:, :, i]
        attn = (q_i @ k_i.transpose(-1, -2)).masked_fill_(mask, 0)
        v_prime = k_cumdecay[:, :, i] @ S
        v_new = v_i - v_prime
        o_inter = q_i @ S
        o[:, :, i] = o_inter + attn @ v_new
        # chunk state update
        S = S + k_i.transpose(-1, -2) @ v_new
        
    final_state = None
    if output_final_state:
        final_state = S

    return rearrange(o, 'b h n c d -> b h (n c) d'), final_state



if __name__ == '__main__':
    beta_mult = 2 # 2 allows for negative eigenvalues but computation is less stable (works with atol=1e-3 and L=4096 DK=512, H=B=1)
    atol = 1e-3
    B = 1
    H = 2
    L = 32
    DK = 512
    DV = 512
    chunk_size=64
    q = (torch.randn(B, H, L, DK)).cuda().requires_grad_(True)
    k = (torch.randn(B, H, L, DK)).cuda()
    k = torch.nn.functional.normalize(k, dim=-1, p=2).requires_grad_(True)
    v = (torch.randn(B, H, L, DV)).cuda().requires_grad_(True)
    beta = (beta_mult*(torch.randn(B, H, L).cuda().sigmoid())).requires_grad_(True)

    o, _ = delta_rule_recurrence(q, k, v, beta)
    do = torch.randn(B, H, L, DV).cuda()
    o.backward(do, retain_graph=True)
    q_grad, q.grad = q.grad, None
    k_grad, k.grad = k.grad, None
    v_grad, v.grad = v.grad, None
    beta_grad, beta.grad = beta.grad, None

    o2, _ = delta_rule_chunkwise(q, k, v, beta, chunk_size=chunk_size)
    o2.backward(do)
    print(f"Testing with tolerance {atol}")
    assert torch.allclose(o, o2, atol=atol), breakpoint()
    assert torch.allclose(q.grad, q_grad, atol=atol), breakpoint()
    assert torch.allclose(k.grad, k_grad, atol=atol), breakpoint()
    assert torch.allclose(v.grad, v_grad, atol=atol), breakpoint()
    assert torch.allclose(beta.grad, beta_grad, atol=atol), breakpoint()
    print("All passed!")
