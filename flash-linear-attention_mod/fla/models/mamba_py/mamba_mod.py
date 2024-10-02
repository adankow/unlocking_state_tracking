import math
from dataclasses import dataclass
from typing import Union

import torch
from fla.models.mamba_py.mamba import Mamba, MambaBlock, RMSNorm, ResidualBlock
from fla.models.mamba_py.pscan import pscan
from torch import nn


@dataclass
class MambaLMConfig:
    d_model: int  #  D
    n_layers: int
    dt_rank: Union[int, str] = 'auto'
    d_state: int = 16  #  N in paper/comments
    expand_factor: int = 2  #  E in paper/comments
    d_conv: int = 4

    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: str = "random"  #  "random" or "constant"
    dt_scale: float = 1.0
    dt_init_floor = 1e-4
    positive_and_negative: bool = True

    rms_norm_eps: float = 1e-5
    base_std: float = 0.02

    bias: bool = False
    conv_bias: bool = True
    inner_layernorms: bool = False  # apply layernorms to internal activations

    mup: bool = False
    mup_base_width: float = 128  # width=d_model

    pscan: bool = True  #  use parallel scan mode or sequential mode when training
    use_cuda: bool = False  # use official CUDA implementation when training (not compatible with (b)float16)

    def __post_init__(self):
        self.d_inner = self.expand_factor * self.d_model  # E*D = ED in comments

        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)

        # muP
        if self.mup:
            self.mup_width_mult = self.d_model / self.mup_base_width


class PositiveNegativeMambaBlock(MambaBlock):
    def __init__(self, config: MambaLMConfig):
        super().__init__(config)
        self.positive_and_negative = config.positive_and_negative

    def selective_scan(self, x, delta, A, B, C, D):
        #  x : (B, L, ED)
        #  Δ : (B, L, ED)
        #  A : (ED, N)
        #  B : (B, L, N)
        #  C : (B, L, N)
        #  D : (ED)

        #  y : (B, L, ED)

        deltaA = torch.exp(delta.unsqueeze(-1) * A)  #  (B, L, ED, N)
        if self.positive_and_negative:
            deltaA = deltaA * 2 - 1
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)  #  (B, L, ED, N)

        BX = deltaB * (x.unsqueeze(-1))  #  (B, L, ED, N)

        hs = pscan(deltaA, BX)

        y = (hs @ C.unsqueeze(-1)).squeeze(3)  #  (B, L, ED, N) @ (B, L, N, 1) -> (B, L, ED, 1)

        y = y + D * x

        return y


class ResidualBlockPositiveNegative(ResidualBlock):
    def __init__(self, config: MambaLMConfig):
        super().__init__(config)

        self.mixer = PositiveNegativeMambaBlock(config)
        self.norm = RMSNorm(config.d_model, config.rms_norm_eps, config.mup)


class MambaLM(Mamba):
    def __init__(self, config: MambaLMConfig, vocab_size, embedding_dim):
        super().__init__(config)
        self.token_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.layers = nn.ModuleList([ResidualBlockPositiveNegative(config) for _ in range(config.n_layers)])

        self.lm_head = nn.Linear(
            in_features=embedding_dim,
            out_features=vocab_size,
            bias=False,
        )

    def forward(self, x):
        x = self.token_embedding(x)
        x = super().forward(x)
        x = self.lm_head(x)
        return x
