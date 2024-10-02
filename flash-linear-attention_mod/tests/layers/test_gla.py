# -*- coding: utf-8 -*-

import pytest
import torch

from fla.layers.gla import GatedLinearAttention


@pytest.mark.parametrize("batch_size", [4, 8])
@pytest.mark.parametrize("seq_len", [1024])
@pytest.mark.parametrize("hidden_size", [2048])
@pytest.mark.parametrize("activation", ['swish'])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_gla(batch_size: int, seq_len: int, hidden_size: int, dtype: torch.dtype, activation: str):
    naive = GatedLinearAttention(hidden_size=hidden_size, gate_fn=activation, fuse_norm=False).to(dtype).cuda()
    fused = GatedLinearAttention(hidden_size=hidden_size, gate_fn=activation, fuse_norm=True).to(dtype).cuda()
    fused.load_state_dict(naive.state_dict())

    x = torch.randn(batch_size, seq_len, hidden_size).to(dtype).cuda()
    naive_x = x.clone().requires_grad_(True)
    fused_x = x.clone().requires_grad_(True)
    naive_o = naive(naive_x)
    fused_o = fused(fused_x)
    naive_o.sum().backward()
    fused_o.sum().backward()
    assert naive_o.allclose(fused_o, 0, 1e-2)
    assert naive_x.grad.allclose(fused_x.grad, 0, 1e-2)

