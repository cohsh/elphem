import numpy as np
from dataclasses import dataclass

from elphem.lattice.rotation import LatticeRotation
from elphem.lattice.lattice_constant import LatticeConstant3D
from elphem.lattice.path import PathValues
from elphem.common.brillouin import SpecialPoints3D

class Cell3D:
    def __init__(self):
        self.basis = None
    
    def calculate_volume(self) -> float:
        """Calculates the volume of the cell.

        Returns:
            float: The volume calculated using the determinant of the basis vectors.
        """
        return np.dot(self.basis[0], np.cross(self.basis[1], self.basis[2]))

class Cell2D:
    def __init__(self):
        self.basis = None
    
    def calculate_volume(self) -> float:
        return np.linalg.norm(np.cross(self.basis[0], self.basis[1]))

class Cell1D:
    def __init__(self):
        self.basis = None
    
    def calculate_volume(self) -> float:
        return self.basis

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
        
        return LatticeRotation3D.optimize(basis, axis)

@dataclass
class ReciprocalCell3D(Cell3D):
    """Defines the reciprocal cell for a crystal based on the lattice constants of the primitive cell."""
    lattice_constant: LatticeConstant3D
    
    def __post_init__(self):
        """Initializes and builds the basis for the reciprocal cell."""
        self.basis = self.build()
        self.volume = self.calculate_volume()
    
    def build(self) -> np.ndarray:
        """Constructs the basis matrix for the reciprocal cell from the primitive cell.

        Returns:
            np.ndarray: The basis matrix of the reciprocal cell.
        """
        primitive_cell = PrimitiveCell3D(self.lattice_constant)

        basis = np.zeros((3,3))
        
        primitive_vector = primitive_cell.basis
        for i in range(3):
            j = (i+1) % 3
            k = (i+2) % 3
            basis[i] = np.cross(primitive_vector[j], primitive_vector[k])

        basis *= 2.0 * np.pi / primitive_cell.volume
        
        return basis

    def get_reciprocal_vectors(self, n_g: int) -> np.ndarray:
        """Generate the reciprocal lattice vectors used to define the Brillouin zone boundaries.

        Returns:
            np.ndarray: An array of reciprocal lattice vectors.
        """

        n_cut = np.ceil(np.cbrt(n_g))

        n_1d = np.arange(-n_cut, n_cut + 1)
        n_3d = np.array(np.meshgrid(n_1d, n_1d, n_1d)).T.reshape(-1, 3)
        
        g = n_3d @ self.basis
        g_norm = np.linalg.norm(g, axis=-1).round(decimals=5)
        g_norm_unique = np.unique(g_norm)

        g_list = []

        for g_ref in g_norm_unique:
            count = 0
            for g_compare in g_norm:
                if g_compare == g_ref:
                    g_list.append(g[count])
                count += 1

        return np.array(g_list[0:n_g])

    def get_monkhorst_pack_grid(self, n_x: int, n_y: int, n_z: int) -> np.ndarray:
        x = (2 * np.arange(1, n_x + 1) - n_x - 1) / (2 * n_x)
        y = (2 * np.arange(1, n_y + 1) - n_y - 1) / (2 * n_y)
        z = (2 * np.arange(1, n_z + 1) - n_z - 1) / (2 * n_z)

        bx = np.broadcast_to(x[:, np.newaxis, np.newaxis], (n_x, n_y, n_z))
        by = np.broadcast_to(y[np.newaxis, :, np.newaxis], (n_x, n_y, n_z))
        bz = np.broadcast_to(z[np.newaxis, np.newaxis, :], (n_x, n_y, n_z))

        aligned_k = np.stack([bx, by, bz], axis=-1).reshape(-1, 3) @ self.basis

        return aligned_k

    def get_path(self, k_names: list[str], n_split: int) -> PathValues:
        """Calculates a path through specified special points in the Brillouin zone.

        Args:
            k_names (list[str]): List of special point names to form the path.
            n (int): Number of points between each special point.

        Returns:
            tuple: Returns the total length of the path, the path coordinates, and the lengths at special points.
        """
        k_via = [self.calculate_special_k(s) for s in k_names]
        n_via = len(k_via) - 1

        major_scales = np.empty((n_via+1,))
        minor_scales = np.empty((n_via * n_split,))
        k = np.empty((n_via * n_split, 3))

        count = 0
        length_part = 0.0
        major_scales[0] = 0.0

        for i in range(n_via):
            direction = k_via[i+1] - k_via[i]
            length = np.linalg.norm(direction)

            x = np.linspace(0.0, 1.0, n_split)
            
            for j in range(n_split):
                k[count] = k_via[i] + x[j] * direction
                minor_scales[count] = x[j] * length + length_part
                count += 1
            
            length_part += length
            major_scales[i+1] = length_part

        k_path = PathValues(major_scales, minor_scales, k)

        return k_path

    def calculate_special_k(self, k_name: str) -> np.ndarray:
        """Retrieves the coordinates of special k-points based on the crystal structure.

        Args:
            k_name (str): The name of the special point.

        Returns:
            np.ndarray: The coordinates of the special point.

        Raises:
            ValueError: If an invalid crystal structure name is provided.
        """
        if self.lattice_constant.crystal_structure == 'bcc':
            special_point = SpecialPoints3D.BCC[k_name]
        elif self.lattice_constant.crystal_structure == 'fcc':
            special_point = SpecialPoints3D.FCC[k_name]
        elif self.lattice_constant.crystal_structure == 'sc':
            special_point = SpecialPoints3D.SC[k_name]
        else:
            raise ValueError("Invalid name specified.")
        
        return np.array(special_point) @ self.basis