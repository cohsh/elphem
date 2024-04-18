import numpy as np
from dataclasses import dataclass

from elphem.lattice.rotation import LatticeRotation
from elphem.const.brillouin import SpecialPoints

@dataclass
class LatticeConstant:
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

class Cell:
    """Base class for a crystal cell, providing basic structure and methods."""
    def __init__(self):
        """Initializes the Cell with a basis matrix."""
        self.basis = self.build()
    
    def build(self) -> np.ndarray:
        """Builds the default identity matrix for the cell basis.

        Returns:
            np.ndarray: A 3x3 identity matrix.
        """
        return np.identity(3)
    
    def volume(self) -> float:
        """Calculates the volume of the cell.

        Returns:
            float: The volume calculated using the determinant of the basis vectors.
        """
        volume = np.dot(self.basis[0], np.cross(self.basis[1], self.basis[2]))
        return volume
    
    def element(self, a: np.ndarray) -> np.ndarray:
        """Transforms a vector a by the basis matrix.

        Args:
            a (np.ndarray): The vector to transform.

        Returns:
            np.ndarray: The transformed vector.
        """
        x = np.dot(a, self.basis)
        return x

    @staticmethod
    def optimize(basis: np.ndarray) -> np.ndarray:
        """Optimizes the cell basis using the LatticeRotation utility.

        Args:
            basis (np.ndarray): The basis matrix to optimize.

        Returns:
            np.ndarray: The optimized basis matrix.
        """
        axis = np.array([1.0] * 3)
        
        basis = LatticeRotation.optimize(basis, axis)
        
        return basis

@dataclass
class PrimitiveCell(Cell):
    """Defines the primitive cell for a crystal based on the lattice constants."""
    lattice_constant: LatticeConstant
    
    def __post_init__(self):
        """Initializes and builds the basis for the primitive cell."""
        self.basis = self.build()
    
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

@dataclass
class ReciprocalCell(Cell):
    """Defines the reciprocal cell for a crystal based on the lattice constants of the primitive cell."""
    lattice_constant: LatticeConstant
    
    def __post_init__(self):
        """Initializes and builds the basis for the reciprocal cell."""
        self.basis = self.build()
    
    def build(self) -> np.ndarray:
        """Constructs the basis matrix for the reciprocal cell from the primitive cell.

        Returns:
            np.ndarray: The basis matrix of the reciprocal cell.
        """
        primitive_cell = PrimitiveCell(self.lattice_constant)

        basis = np.zeros((3,3))
        
        primitive_vector = primitive_cell.basis
        for i in range(3):
            j = (i+1) % 3
            k = (i+2) % 3
            basis[i] = np.cross(primitive_vector[j], primitive_vector[k])

        basis *= 2.0 * np.pi / primitive_cell.volume()
        
        return basis
    
    def simple_grid(self, n_x: int, n_y: int, n_z: int) -> np.ndarray:
        """Generate simple k-grid

        Args:
            n_x (int): Number of mesh
            n_y (int): Number of mesh
            n_z (int): Number of mesh
        
        Returns:
            k (np.ndarray): k-grid
        """

        k_x = np.linspace(-0.5, 0.5, n_x)
        k_y = np.linspace(-0.5, 0.5, n_y)
        k_z = np.linspace(-0.5, 0.5, n_z)
        k = np.array(np.meshgrid(k_x, k_y, k_z, indexing='ij')).T.reshape(-1, 3) @ self.basis

        return k

    def monkhorst_pack_grid(self, n_x: int, n_y: int, n_z: int) -> np.ndarray:
        """Generate Monkhorst and Pack k-grid

        Args:
            n_x (int): Number of mesh
            n_y (int): Number of mesh
            n_z (int): Number of mesh
        
        Returns:
            k (np.ndarray): k-grid
        """

        r_x = np.arange(1, n_x + 1)
        r_y = np.arange(1, n_y + 1)
        r_z = np.arange(1, n_z + 1)

        k_x = (2 * r_x - n_x - 1) / ( 2 * n_x )
        k_y = (2 * r_y - n_y - 1) / ( 2 * n_y )
        k_z = (2 * r_z - n_z - 1) / ( 2 * n_z )
        k = np.array(np.meshgrid(k_x, k_y, k_z, indexing='ij')).T.reshape(-1, 3) @ self.basis

        return k

    def path(self, k_names: list[str], n: int) -> np.ndarray:
        """Calculates a path through specified special points in the Brillouin zone.

        Args:
            k_names (list[str]): List of special point names to form the path.
            n (int): Number of points between each special point.

        Returns:
            tuple: Returns the total length of the path, the path coordinates, and the lengths at special points.
        """
        k_via = [self.get_special_k(s) for s in k_names]
        n_via = len(k_via) - 1

        total_length = np.empty((n_via * n,))
        special_length = np.empty((n_via+1,))
        k = np.empty((n_via * n, 3))

        count = 0
        length_part = 0.0
        special_length[0] = 0.0

        for i in range(n_via):
            direction = (np.array(k_via[i+1]) - np.array(k_via[i])) @ self.basis
            length = np.linalg.norm(direction)

            x = np.linspace(0.0, 1.0, n)
                        
            for j in range(n):
                k[count] = k_via[i] @ self.basis + x[j] * direction
                total_length[count] = x[j] * length + length_part
                count += 1
            
            length_part += length
            special_length[i+1] = length_part

        return total_length, k, special_length

    def get_special_k(self, k_name: str) -> np.ndarray:
        """Retrieves the coordinates of special k-points based on the crystal structure.

        Args:
            k_name (str): The name of the special point.

        Returns:
            np.ndarray: The coordinates of the special point.

        Raises:
            ValueError: If an invalid crystal structure name is provided.
        """
        if self.lattice_constant.crystal_structure == 'bcc':
            return SpecialPoints.BCC[k_name]
        elif self.lattice_constant.crystal_structure == 'fcc':
            return SpecialPoints.FCC[k_name]
        elif self.lattice_constant.crystal_structure == 'sc':
            return SpecialPoints.SC[k_name]
        else:
            raise ValueError("Invalid name specified.")

@dataclass
class EmptyLattice:
    """A class to simulate an empty lattice for a given crystal structure and lattice constant a."""
    crystal_structure: str
    a: float

    def __post_init__(self):
        """Initializes the lattice constants, primitive, and reciprocal cells along with their volumes and bases."""
        self.constants = self.get_lattice_constant()
        self.primitive_cell = PrimitiveCell(self.constants)
        self.reciprocal_cell = ReciprocalCell(self.constants)

        self.volume = {
            "primitive": self.primitive_cell.volume(),
            "reciprocal": self.reciprocal_cell.volume()
        }
        self.basis = {
            "primitive": self.primitive_cell.basis,
            "reciprocal": self.reciprocal_cell.basis
        }

    def get_lattice_constant(self) -> LatticeConstant:
        """Determines lattice constants based on the crystal structure.

        Returns:
            LatticeConstant: The lattice constants and angles for the specified crystal structure.

        Raises:
            ValueError: If an invalid crystal structure name is specified.
        """
        crystal_structure_lower = self.crystal_structure.lower()

        alpha_values = {
            'bcc': 109.47,
            'fcc': 60.0,
            'sc': 90.0
        }

        alpha = alpha_values.get(crystal_structure_lower)

        if alpha is not None:
            return LatticeConstant(self.a, self.a, self.a, alpha, alpha, alpha, crystal_structure_lower)
        else:
            raise ValueError("Invalid crystal structure specified.")

@dataclass
class Grid:
    basis: np.ndarray
    mesh: np.ndarray

class SimpleGrid(Grid):
    def __init__(self):
#        super().__init__()


    def generate(self, n_x: int, n_y: int, n_z: int) -> np.ndarray:
        """Generate simple k-grid

        Args:
            n_x (int): Number of mesh
            n_y (int): Number of mesh
            n_z (int): Number of mesh
        
        Returns:
            k (np.ndarray): k-grid
        """

        k_x = np.linspace(-0.5, 0.5, n_x)
        k_y = np.linspace(-0.5, 0.5, n_y)
        k_z = np.linspace(-0.5, 0.5, n_z)
        k = np.array(np.meshgrid(k_x, k_y, k_z, indexing='ij')).T.reshape(-1, 3) @ self.basis

        return k

    def monkhorst_pack_grid(self, n_x: int, n_y: int, n_z: int) -> np.ndarray:
        """Generate Monkhorst and Pack k-grid

        Args:
            n_x (int): Number of mesh
            n_y (int): Number of mesh
            n_z (int): Number of mesh
        
        Returns:
            k (np.ndarray): k-grid
        """

        r_x = np.arange(1, n_x + 1)
        r_y = np.arange(1, n_y + 1)
        r_z = np.arange(1, n_z + 1)

        k_x = (2 * r_x - n_x - 1) / ( 2 * n_x )
        k_y = (2 * r_y - n_y - 1) / ( 2 * n_y )
        k_z = (2 * r_z - n_z - 1) / ( 2 * n_z )
        k = np.array(np.meshgrid(k_x, k_y, k_z, indexing='ij')).T.reshape(-1, 3) @ self.basis

        return k