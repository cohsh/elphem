from dataclasses import dataclass
import numpy as np

@dataclass
class BrillouinPathValues:
    distances: np.ndarray
    values: np.ndarray
    special_distances: np.ndarray