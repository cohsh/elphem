import numpy as np
from dataclasses import dataclass

from elphem.lattice.rotation import LatticeRotation3D, LatticeRotation2D
from elphem.lattice.cell import Cell1D, Cell2D, Cell3D
from elphem.lattice.lattice_constant import LatticeConstant3D

@dataclass
class PrimitiveCell3D(Cell3D):
    """Defines the primitive cell for a crystal based on the lattice constants."""
    lattice_constant: LatticeConstant3D
    
    def __post_init__(self):
        """Initializes and builds the basis for the primitive cell."""
        self.basis = self.build()
        self.volume = self.calculate_volume()
    
    def build(self) -> np.ndarray:
        """Constructs the basis matrix for the primitive cell from lattice constants and angles.

        Returns:
            np.ndarray: The basis matrix of the primitive cell.
        """
        length = self.lattice_constant.length
        angle = self.lattice_constant.angle

        basis = np.zeros((3,3))

        basis[0][0] = length[0]
        basis[1][0] = length[1] * np.cos(angle[2])
        basis[1][1] = length[1] * np.sin(angle[2])
        basis[2][0] = length[2] * np.cos(angle[1])
        basis[2][1] = length[2] * (np.cos(angle[0]) - np.cos(angle[1]) * np.cos(angle[2])) / np.sin(angle[2])
        basis[2][2] = np.sqrt(length[2] ** 2 - np.sum(basis[2]**2))

        basis = self.optimize(basis)
        
        return basis

    @staticmethod
    def optimize(basis: np.ndarray) -> np.ndarray:
        """Optimizes the cell basis using the LatticeRotation utility.

        Args:
            basis (np.ndarray): The basis matrix to optimize.

        Returns:
            np.ndarray: The optimized basis matrix.
        """
        axis = np.array([1.0] * 3)
        
        return LatticeRotation3D.calculate_optimized_basis(basis, axis)

@dataclass
class PrimitiveCell2D(Cell2D):
    lattice_constant: LatticeRotation2D

    def __post_init__(self):
        """Initializes and builds the basis for the primitive cell."""
        self.basis = self.build()
        self.volume = self.calculate_volume()
    
    def build(self) -> np.ndarray:
        """Constructs the basis matrix for the primitive cell from lattice constants and angles.

        Returns:
            np.ndarray: The basis matrix of the primitive cell.
        """
        length = self.lattice_constant.length
        angle = self.lattice_constant.angle

        basis = np.zeros((2,2))

        basis[0][0] = length[0]
        basis[1][0] = length[1] * np.cos(angle)
        basis[1][1] = length[1] * np.sin(angle)

        basis = self.optimize(basis)
        
        return basis

    @staticmethod
    def optimize(basis: np.ndarray) -> np.ndarray:
        """Optimizes the cell basis using the LatticeRotation utility.

        Args:
            basis (np.ndarray): The basis matrix to optimize.

        Returns:
            np.ndarray: The optimized basis matrix.
        """
        axis = np.array([1.0, 1.0])
        
        return LatticeRotation2D.calculate_optimized_basis(basis, axis)