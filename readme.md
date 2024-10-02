# Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues

## Abstract:
> Linear Recurrent Neural Networks (LRNNs), such as Mamba, RWKV, GLA, mLSTM, and DeltaNet have emerged as efficient alternatives to transformers in large language modeling, offering linear scaling with sequence length and improved training efficiency. However, LRNNs struggle with state-tracking which is important for, e.g., code comprehension and tracking chess pieces across a board. Even parity, the simplest state-tracking task, which non-linear RNNs like LSTMs handle effectively, cannot be solved by current LRNNs. Recently, \citet{sarrof2024expressive} demonstrated that the failure of LRNNs like Mamba to solve parity stems from restricting the eigenvalue range of their diagonal state-transition matrices to $[0, 1]$, and that incorporating negative eigenvalues can resolve this issue. We generalize this result to full matrix LRNNs, which have recently shown promise in models such as DeltaNet. We prove that no finite-precision LRNN with state-transition matrices having only positive eigenvalues can solve parity, while complex eigenvalues are needed to count modulo $3$. Notably, we also prove that LRNNs can learn any regular language when their state-transition matrices are products of identity plus vector outer product matrices with eigenvalues in the range $[-1, 1]$. Our empirical results confirm that extending the eigenvalue range of models like Mamba and DeltaNet to include negative values not only enables them to solve parity but consistently improves their performance on state-tracking tasks. Furthermore, pre-training LRNNs with an extended eigenvalue range for language modeling achieves comparable performance and stability while showing promise for coding tasks. Our work enhances the expressivity of modern LRNNs, broadening their applicability without changing the cost of training or inference.



## Requirements
- Python >= 3.11
## Dependencies
```
$ pip install -r requirements.in
```
## Experiments
We provide all the code necessary to reproduce the experiments in the chomsky hierarchy as implemented by xlstm. This is a composition of repositories. It contains the flash-linear attention library with extended eigenvalue range for DeltaNet and Mamba with similar modifications but for CUDA code of the associative scan.

To install, please first go to the `flash-linear-attention` directory and run the following command:
```
$ pip install -e .
```
Then, go to the `mamba_dev` directory and run the following command:
```
$ pip install -e .
```
To run the experiments please go to xlstm/experiments
### Chomsky Hierarchy
#### Parity
mLSTM:
```
$ PYTHONPATH=$PWD python experiments/main.py --config=experiments/parity_xlstm10.yaml 
```
sLSTM:
```
$ PYTHONPATH=$PWD python experiments/main.py --config=experiments/parity_xlstm01.yaml 
```
---
Mamba [0, 1]:
```
$ PYTHONPATH=$PWD python experiments/main.py --config=experiments/parity_mamba.yaml 
```
Mamba [-1, 1]:
```
$ PYTHONPATH=$PWD python experiments/main.py --config=experiments/parity_mamba_2.yaml 
```
---
DeltaNet [0, 1]:
```
$ PYTHONPATH=$PWD python experiments/main.py --config=experiments/parity_delta_net.yaml 
```
DeltaNet [-1, 1]:
```
$ PYTHONPATH=$PWD python experiments/main.py --config=experiments/parity_delta_net_2.yaml 
```

#### Modular Arithmetic w/o brackets 
mLSTM:
```
$ PYTHONPATH=$PWD python experiments/main.py --config=experiments/mod_arithmetic_xlstm10.yaml 
```
sLSTM:
```
$ PYTHONPATH=$PWD python experiments/main.py --config=experiments/mod_arithmetic_xlstm01.yaml 
```
---
Mamba [0, 1]:
```
$ PYTHONPATH=$PWD python experiments/main.py --config=experiments/mod_arithmetic_mamba.yaml 
```
Mamba [-1, 1]:
```
$ PYTHONPATH=$PWD python experiments/main.py --config=experiments/mod_arithmetic_mamba_2.yaml 
```
---
DeltaNet [0, 1]:
```
$ PYTHONPATH=$PWD python experiments/main.py --config=experiments/mod_arithmetic_delta_net.yaml 
```
DeltaNet [-1, 1]:
```
$ PYTHONPATH=$PWD python experiments/main.py --config=experiments/mod_arithmetic_delta_net_2.yaml 
```


#### Modular Arithmetic w/ brackets 
mLSTM:
```
$ PYTHONPATH=$PWD python experiments/main.py --config=experiments/mod_arithmetic_with_brackets_xlstm10.yaml 
```
sLSTM:
```
$ PYTHONPATH=$PWD python experiments/main.py --config=experiments/mod_arithmetic_with_brackets_xlstm01.yaml 
```
---
Mamba [0, 1]:
```
$ PYTHONPATH=$PWD python experiments/main.py --config=experiments/mod_arithmetic_with_brackets_mamba.yaml 
```
Mamba [-1, 1]:
```
$ PYTHONPATH=$PWD python experiments/main.py --config=experiments/mod_arithmetic_with_brackets_mamba_2.yaml 
```
---
DeltaNet [0, 1]:
```
$ PYTHONPATH=$PWD python experiments/main.py --config=experiments/mod_arithmetic_with_brackets_delta_net.yaml 
```
DeltaNet [-1, 1]:
```
$ PYTHONPATH=$PWD python experiments/main.py --config=experiments/mod_arithmetic_with_brackets_delta_net_2.yaml 
```