from dataclasses import dataclass
import numpy as np

@dataclass
class PathValues:
    major_scales: np.ndarray
    minor_scales: np.ndarray
    values: np.ndarray
    
    def derive(self, values: np.ndarray) -> 'PathValues':
        """Create a new PathValues object with the same scales but new values."""
        return PathValues(self.major_scales, self.minor_scales, values)