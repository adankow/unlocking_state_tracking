# -*- coding: utf-8 -*-

import pytest
import torch

from fla.layers.delta_net import DeltaNet


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("seq_len", [2048])
@pytest.mark.parametrize("hidden_size", [2048])
@pytest.mark.parametrize("sigmoid_scale", [1., 2.])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_gla(batch_size: int, seq_len: int, hidden_size: int, sigmoid_scale: int, dtype: torch.dtype):
    chunk_size=64
    atol=1e-2 # since initialization is random we can keep this tolerance even with sigmoid_scale = 2.

    x = torch.randn(batch_size, seq_len, hidden_size).to(dtype).cuda()    
    
    for mode in ['naive', 'chunk', 'fused_recurrent', 'fused_chunk', 'naive_chunk']:
        model = DeltaNet(hidden_size=hidden_size, sigmoid_scale=sigmoid_scale, mode=mode, chunk_size=chunk_size).to(dtype).cuda()
        
        if mode == 'naive':
            params_saved = model.state_dict()
        else:
            model.load_state_dict(params_saved)

        m_x = x.clone().requires_grad_(True)
        m_out, _, _ = model(m_x)
        m_out.sum().backward()
        m_grad = m_x.grad
        if mode == 'naive':
            ref_out = m_out
            ref_grad = m_grad
        else:
            assert torch.allclose(ref_out, m_out, atol=atol), f"top 4 abs diff = {torch.topk((ref_out - m_out).abs().flatten(), 4)}"
            assert torch.allclose(ref_grad, m_grad, atol=atol), f"top 4 abs diff = {torch.topk((ref_grad - m_grad).abs().flatten(), 4)}"
            print(f'passed for mode={mode}')


if __name__ == '__main__':
    test_gla(batch_size=1, seq_len=1024, hidden_size=512, sigmoid_scale=1., dtype=torch.bfloat16)
    print("Test passed!")

