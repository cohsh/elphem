from dataclasses import dataclass
import numpy as np

@dataclass
class PathValues:
    """A class for values with path.
    
    Attributes:
        major_scales (np.ndarray): A numpy array of major scales of path.
        minor_scales (np.ndarray): A numpy array of minor scales of path.
        values (np.ndarray): A numpy array of values.
    """
    major_scales: np.ndarray
    minor_scales: np.ndarray
    values: np.ndarray