import torch
from fla.models.mamba_py.mamba_mod import MambaLMConfig, MambaLM

# Define the configuration
config = MambaLMConfig(
    d_model=128,
    n_layers=4,
    dt_rank='auto',
    d_state=16,
    expand_factor=2,
    d_conv=4,
    positive_and_negative=True
)

# Set up model parameters
vocab_size = 10000
embedding_dim = 128
positive_and_negative = True

# Create the model
model = MambaLM(config, vocab_size, embedding_dim)

# Create a sample input
batch_size = 2
sequence_length = 10
input_tensor = torch.randint(0, vocab_size, (batch_size, sequence_length))

# Run a forward pass
with torch.no_grad():
    output = model(input_tensor)

print(f"Input shape: {input_tensor.shape}")
print(f"Output shape: {output.shape}")
