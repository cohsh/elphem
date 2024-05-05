import numpy as np
from dataclasses import dataclass

@dataclass
class LatticeConstant3D:
    """Defines the lattice constants and angles for a crystal structure.

    Attributes:
        a (float): Length of the first lattice vector.
        b (float): Length of the second lattice vector.
        c (float): Length of the third lattice vector.
        alpha (float): Angle between b and c lattice vectors.
        beta (float): Angle between a and c lattice vectors.
        gamma (float): Angle between a and b lattice vectors.
        crystal_structure (str): The type of crystal structure (e.g., 'bcc', 'fcc', 'sc').
    """
    a: float
    b: float
    c: float
    alpha: float
    beta: float
    gamma: float
    crystal_structure: str
    
    def __post_init__(self):
        """Converts angles to radians and stores lengths and angles as numpy arrays."""
        self.length = np.array([self.a, self.b, self.c])
        self.angle = np.radians(np.array([self.alpha, self.beta, self.gamma]))
        
    def rescale(self, factor: float) -> None:
        """Rescales the lattice constants by a given factor.

        Args:
            factor (float): The factor by which the lattice lengths are multiplied.
        """
        self.length *= factor

@dataclass
class LatticeConstant2D:
    """Defines the lattice constants and angles for a crystal structure.

    Attributes:
        a (float): Length of the first lattice vector.
        b (float): Length of the second lattice vector.
        c (float): Length of the third lattice vector.
        alpha (float): Angle between b and c lattice vectors.
        beta (float): Angle between a and c lattice vectors.
        gamma (float): Angle between a and b lattice vectors.
        crystal_structure (str): The type of crystal structure (e.g., 'bcc', 'fcc', 'sc').
    """
    a: float
    b: float
    alpha: float
    crystal_structure: str
    
    def __post_init__(self):
        """Converts angles to radians and stores lengths and angles as numpy arrays."""
        self.length = np.array([self.a, self.b])
        self.angle = np.radians(np.array([self.alpha]))
        
    def rescale(self, factor: float) -> None:
        """Rescales the lattice constants by a given factor.

        Args:
            factor (float): The factor by which the lattice lengths are multiplied.
        """
        self.length *= factor

@dataclass
class LatticeConstant1D:
    """Defines the lattice constants and angles for a crystal structure.

    Attributes:
        a (float): Length of the first lattice vector.
        b (float): Length of the second lattice vector.
        c (float): Length of the third lattice vector.
        alpha (float): Angle between b and c lattice vectors.
        beta (float): Angle between a and c lattice vectors.
        gamma (float): Angle between a and b lattice vectors.
        crystal_structure (str): The type of crystal structure (e.g., 'bcc', 'fcc', 'sc').
    """
    a: float
    crystal_structure: str
    
    def __post_init__(self):
        """Converts angles to radians and stores lengths and angles as numpy arrays."""
        self.length = np.array([self.a])
        
    def rescale(self, factor: float) -> None:
        """Rescales the lattice constants by a given factor.

        Args:
            factor (float): The factor by which the lattice lengths are multiplied.
        """
        self.length *= factor