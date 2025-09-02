import os
import random
import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)

def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Enforce deterministic behavior in cuDNN (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # PyTorch >= 1.8
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        logger.warning("Deterministic algorithms not fully supported in this PyTorch version.")

    os.environ["PYTHONHASHSEED"] = str(seed)

    logger.info(f"Random seed set to {seed}")
