#!/bin/bash
set -e


pip install causal-conv1d>=1.4.0
pip install dacite
pip install omegaconf
pip install torchmetrics

cd flash-linear-attention_mod/; pip install -e .
cd ../mamba_dev/; pip install -e . --no-cache-dir