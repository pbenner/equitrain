###########################################################################################
# Tools for torch
# Authors: Ilyes Batatia, Gregor Simm
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import numpy as np
import torch

def to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.cpu().detach().numpy()

def set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
