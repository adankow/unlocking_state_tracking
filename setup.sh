#!\bin\bash

pip install causal-conv1d>=1.4.0
pip install dacite
pip install omegaconf
pip install torchmetrics

cd unlocking_state_tracking/flash-linear-attention_mod/; pip install -e .
cd unlocking_state_tracking/mamba_dev/; pip install -e . --no-cache-dir