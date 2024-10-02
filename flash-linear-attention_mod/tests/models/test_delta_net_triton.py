import torch
from torch import nn
import pytest
from fla.models.delta_net import DeltaNetConfig, DeltaNetModel

def test_deltanet_forward_and_backward_pass():
    # Set up a simple config and initialize the model
    config = DeltaNetConfig(
        vocab_size=100,       # smaller vocab for testing
        hidden_size=64,       # small hidden size for testing
        num_hidden_layers=2,  # fewer layers for a quick test
        num_heads=4,          # reduced number of heads for testing
        use_short_conv=True,
        fuse_cross_entropy=True,
        attn_mode="gla_mod_chunk",
        chunk_size=32,
        sigmoid_scale=2.0
    )
    
    model = DeltaNetModel(config).cuda().to(torch.bfloat16)
    
    # Define input parameters
    batch_size = 2
    seq_length = 256
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length)).cuda()
    
    # Run forward pass
    output = model(input_ids=input_ids)
        
    # Check output shape
    assert output.last_hidden_state.shape == (batch_size, seq_length, config.hidden_size), \
        f"Unexpected output shape: {output.last_hidden_state.shape}"

    # Test caching mechanism (past_key_values)
    output_with_cache = model(input_ids=input_ids, use_cache=True)
    assert output_with_cache.past_key_values is not None, "Expected cache to be returned."
    
    # Define a simple dummy loss function (mean of the output)
    dummy_target = torch.randn(batch_size, seq_length, config.hidden_size).cuda().to(torch.bfloat16)  # random target with same shape as output
    loss_fn = nn.MSELoss()  # Mean Squared Error loss for simplicity
    loss = loss_fn(output.last_hidden_state, dummy_target)

    # Perform backward pass
    loss.backward()

    # Check that gradients are not None for model parameters
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradients not found for parameter: {name}"

    # Test caching mechanism (past_key_values)
    output_with_cache = model(input_ids=input_ids, use_cache=True)
    assert output_with_cache.past_key_values is not None, "Expected cache to be returned."



if __name__ == "__main__":
    test_deltanet_forward_and_backward_pass()
    print("Test passed")
    #pytest.main()