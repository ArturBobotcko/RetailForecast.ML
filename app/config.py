import random

import numpy as np

DEFAULT_VALIDATION_FRACTION = 0.2
GLOBAL_RANDOM_SEED = 42


def set_all_seeds(seed: int = GLOBAL_RANDOM_SEED) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
