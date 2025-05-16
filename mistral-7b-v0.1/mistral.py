import math
import torch
import torch.nn as nn
from typing import Tuple

from config import MISTRAL_7B_V0_1_CONFIG as CONFIG
from rope import RoPE

# RMSNorm


# GQA with SWA


# FFN


# Decoder Block with Residuals


# Transformer with stacked Decoder Blocks, final RMSNorm, final Linear layer (LM Head), Softmax
class Mistral(nn.Module):
    pass