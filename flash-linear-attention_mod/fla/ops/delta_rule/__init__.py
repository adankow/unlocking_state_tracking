# -*- coding: utf-8 -*-

from .chunk import chunk_delta_rule
from .chunk_fuse import fused_chunk_delta_rule
from .recurrent_fuse import fused_recurrent_delta_rule
from .naive_compatible import delta_rule_recurrence, delta_rule_chunkwise
from .naive_gla import gla_mod_recurrent, gla_mod_chunk

__all__ = [
    'fused_chunk_delta_rule',
    'fused_recurrent_delta_rule',
    'chunk_delta_rule',
    'delta_rule_recurrence',
    'delta_rule_chunkwise',
    'gla_mod_recurrent',
    'gla_mod_chunk',
]
