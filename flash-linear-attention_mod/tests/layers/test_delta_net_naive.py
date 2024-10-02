# -*- coding: utf-8 -*-

import pytest
import torch

from fla.layers.delta_net_no_triton import DeltaNetNoTriton


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("seq_len", [2048])
@pytest.mark.parametrize("hidden_size", [2048])
@pytest.mark.parametrize("sigmoid_scale", [1., 2.])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_gla(batch_size: int, seq_len: int, hidden_size: int, sigmoid_scale: int, dtype: torch.dtype):
    naive = DeltaNetNoTriton(hidden_size=hidden_size, sigmoid_scale=sigmoid_scale, mode='naive', chunk_size=10, use_short_conv=False).to(dtype).cuda()
    naive_chunk = DeltaNetNoTriton(hidden_size=hidden_size, sigmoid_scale=sigmoid_scale, mode='naive_chunk', chunk_size=10, use_short_conv=False).to(dtype).cuda()
    naive.load_state_dict(naive_chunk.state_dict())

    atol=1e-4 # since initialization is random we can keep this tolerance even with sigmoid_scale = 2.


    x = torch.randn(batch_size, seq_len, hidden_size).to(dtype).cuda()
    naive_x = x.clone().requires_grad_(True)
    naive_chunk_x = x.clone().requires_grad_(True)

    naive_o, _, _ = naive(naive_x)
    naive_chunk_o, _, _ = naive_chunk(naive_chunk_x)
    naive_o.sum().backward()
    naive_chunk_o.sum().backward()
    

    assert torch.allclose(naive_chunk_o, naive_o, atol=atol), f"top 4 abs diff = {torch.topk((naive_chunk_o - naive_o).abs().flatten(), 4)}"
    assert torch.allclose(naive_chunk_x.grad, naive_x.grad, atol=atol), f"top 4 abs diff = {torch.topk((naive_chunk_x.grad - naive_x.grad).abs().flatten(), 4)}"

if __name__ == '__main__':
    test_gla(batch_size=1, seq_len=40, hidden_size=2048, sigmoid_scale=2, dtype=torch.float32)
    print("Test passed!")

