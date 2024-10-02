# -*- coding: utf-8 -*-

import torch
from einops import rearrange

def gla_mod_recurrent(
    q,
    k,
    v,
    beta,
    initial_state=None,
    output_final_state=False,
):
    orig_dtype = q.dtype
    q, k, v, beta = map(lambda x: x.float(), (q, k, v, beta))
    batch_size, n_heads, seq_len, d_head_k = q.shape
    _, _, _, d_head_v = v.shape
    h = torch.zeros(batch_size, n_heads, d_head_k, d_head_v, dtype=torch.float32, device=q.device)
    o = torch.zeros_like(v)
    scale = d_head_k ** -0.5

    if initial_state is not None:
        h += initial_state

    for i in range(seq_len):
        q_i = q[:, :, i, :] * scale
        k_i = k[:, :, i]
        v_i = v[:, :, i, :]
        beta_i = beta[:, :, i]
        kv_i = k_i[..., None] * v_i[..., None, :]
        h = h * (1 - beta_i[..., None, None]) + kv_i
        o_i = (q_i[..., None] * h).sum(-2)
        o[:, :, i, :] = o_i

    output_state = None
    if output_final_state:
        output_state = h
        
    return o.to(orig_dtype), output_state

def gla_mod_chunk(q, k, v, beta, chunk_size=64,
                  initial_state=None, output_final_state=False,):
    l = beta.shape[-2]
    if chunk_size > l:
        chunk_size = l
    
    assert l % chunk_size == 0    
    
    q, k, v, beta = map(lambda x: x.float(), (q, k, v, beta))
    scale = (q.shape[-1] ** -0.5)
    q = rearrange(q, 'b h (n c) d -> b h n c d', c=chunk_size) * scale
    k = rearrange(k, 'b h (n c) d -> b h n c d', c=chunk_size)
    v = rearrange(v, 'b h (n c) d -> b h n c d', c=chunk_size)
    beta = rearrange(beta, 'b h (n c) -> b h n c', c=chunk_size)
    gamma = 1 - beta
    gamma = gamma.cumprod(-1)
    kv = k.transpose(-1, -2) @ (v *  (gamma[:, :, :, -1, None]/gamma)[..., None])
    
    S = torch.zeros_like(kv)
    if initial_state is not None:
        S[:,:,0] = initial_state


    for i in range(1, gamma.shape[-2]):
        S[:, :, i] = S[:, :, i-1].clone() * gamma[:, :, i-1, -1, None, None] + kv[:, :, i-1]

    inter = (q * gamma[..., None]) @ S
    attn = q @ k.transpose(-1, -2)
    attn = attn * gamma[..., None]/gamma[..., None]
    attn = attn.masked_fill(torch.triu(torch.ones(chunk_size, chunk_size, dtype=bool, device=q.device), diagonal=1), 0)
    intra = attn @ v
    o = inter + intra
    
    final_state = None
    if output_final_state:
        raise NotImplementedError
    
    return rearrange(o, 'b h n c d -> b h (n c) d'), final_state

def stable_division(t1, t2):
    if isinstance(t1, torch.Tensor) and isinstance(t2, torch.Tensor):
        return torch.exp(t1-t2)
    elif isinstance(t1, tuple) and isinstance(t2, tuple) and len(t1) == 2 and len(t2)==2:
        t1_abs_log, t1_sign = t1[0], t1[1]
        t2_abs_log, t2_sign = t2[0], t2[1]

        
        return torch.exp(t1_abs_log - t2_abs_log) * t1_sign/t2_sign
    else:
        NotImplementedError
 

def gla_mod_chunk_log_space(q, k, v, beta, chunk_size=64,
                  initial_state=None, output_final_state=False,):
    l = beta.shape[-2]
    if chunk_size > l:
        chunk_size = l
    
    assert l % chunk_size == 0    
    
    q, k, v, beta = map(lambda x: x.float(), (q, k, v, beta))
    scale = (q.shape[-1] ** -0.5)
    q = rearrange(q, 'b h (n c) d -> b h n c d', c=chunk_size) * scale
    k = rearrange(k, 'b h (n c) d -> b h n c d', c=chunk_size)
    v = rearrange(v, 'b h (n c) d -> b h n c d', c=chunk_size)
    beta = rearrange(beta, 'b h (n c) -> b h n c', c=chunk_size)
    gamma = 1 - beta

    gamma_abs_log = gamma.abs().log()
    gamma_sign = torch.sign(gamma)
    gamma_abs_log = gamma_abs_log.cumsum(-1)
    # gamma_sign = gamma_sign.cumprod(-1)
    gamma_sign = (gamma_sign == -1).to(torch.bool).cumsum(dim=-1)
    gamma_sign = torch.where( gamma_sign == 1, -1, 1).float()
    gamma = (gamma_abs_log, gamma_sign)
    
    gamma_div_1 = stable_division((gamma[0][:, :, :, -1, None], gamma[1]), (gamma[0][:, :, :, -1, None], gamma[1]))[..., None]
    kv = k.transpose(-1, -2) @ (v *  gamma_div_1)
    
    S = torch.zeros_like(kv)
    if initial_state is not None:
        S[:,:,0] = initial_state

    for i in range(1, gamma[0].shape[-2]):
        S[:, :, i] = S[:, :, i-1].clone() * gamma[1][:, :, i-1, -1, None, None] * torch.exp(gamma[0][:, :, i-1, -1, None, None]) + kv[:, :, i-1]

    inter = (q * gamma_sign[..., None] * torch.exp(gamma_abs_log[..., None])) @ S
    attn = q @ k.transpose(-1,-2)
    gamma_div_2 = stable_division((gamma[0][..., None], gamma[1][..., None]), (gamma[0][..., None], gamma[1][..., None]))
    attn = attn * gamma_div_2
    attn = attn.masked_fill(torch.triu(torch.ones(chunk_size, chunk_size, dtype=bool, device=q.device), diagonal=1), 0)
    intra = attn @ v
    o = inter + intra
    
    final_state = None
    if output_final_state:
        raise NotImplementedError
    
    return rearrange(o, 'b h n c d -> b h (n c) d'), final_state



if __name__ == '__main__':
    beta_mult = 1. # 2 allows for negative eigenvalues but computation is less stable (works with atol=1e-3 and L=4096 DK=512, H=B=1)
    B = 10
    H = 1
    L = 2048
    DK = 256
    DV = 16
    chunk_size=32
    print(f"testing with beta_mult = {beta_mult}")
    q = (torch.randn(B, H, L, DK)).cuda().requires_grad_(True)
    k = (torch.randn(B, H, L, DK)).cuda()
    k = torch.nn.functional.normalize(k, dim=-1, p=2).requires_grad_(True)
    v = (torch.randn(B, H, L, DV)).cuda().requires_grad_(True)
    beta = (beta_mult*(torch.randn(B, H, L).cuda().sigmoid())).requires_grad_(True)

    o, _ = gla_mod_recurrent(q, k, v, beta)
    do = torch.randn(B, H, L, DV).cuda()
    o.backward(do)
    q_grad, q.grad = q.grad.clone(), None
    k_grad, k.grad = k.grad.clone(), None
    v_grad, v.grad = v.grad.clone(), None
    beta_grad, beta.grad = beta.grad.clone(), None


    for m in [gla_mod_chunk_log_space, gla_mod_chunk]:
        print(f"Testing {m}")

        # Create CUDA events to measure time
        start_fwd = torch.cuda.Event(enable_timing=True)
        end_fwd = torch.cuda.Event(enable_timing=True)
        start_bwd = torch.cuda.Event(enable_timing=True)
        end_bwd = torch.cuda.Event(enable_timing=True)

        # Forward pass timing
        start_fwd.record()
        o2, _ = m(q, k, v, beta, chunk_size=chunk_size)
        end_fwd.record()
        torch.cuda.synchronize()  # Wait for the events to be recorded
        fwd_time = start_fwd.elapsed_time(end_fwd)  # Time in milliseconds

        # Backward pass timing
        start_bwd.record()
        o2.backward(do)
        torch.cuda.synchronize()
        end_bwd.record()
        bwd_time = start_bwd.elapsed_time(end_bwd)  # Time in milliseconds

        print(f"Forward pass time: {fwd_time:.2f} ms")
        print(f"Backward pass time: {bwd_time:.2f} ms")



        diffs = {
            "output": torch.abs(o - o2),
            "q.grad": torch.abs(q.grad - q_grad),
            "k.grad": torch.abs(k.grad - k_grad),
            "v.grad": torch.abs(v.grad - v_grad),
            "beta.grad": torch.abs(beta.grad - beta_grad)
        }

        for name, diff in diffs.items():
            print(f"{name} -> Max: {torch.max(diff)}, Mean: {torch.mean(diff)}, Std: {torch.std(diff)}")

        
        # reset grads
        q.grad = None
        k.grad = None
        v.grad = None
        beta.grad = None

