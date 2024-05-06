import numpy as np
from dataclasses import dataclass

from elphem.lattice.cell import Cell3D, Cell2D, Cell1D
from elphem.lattice.primitive_cell import PrimitiveCell3D, PrimitiveCell2D, PrimitiveCell1D
from elphem.lattice.lattice_constant import LatticeConstant3D, LatticeConstant2D, LatticeConstant1D
from elphem.lattice.path import PathValues
from elphem.common.brillouin import SpecialPoints3D, SpecialPoints2D, SpecialPoints1D

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

@dataclass
class ReciprocalCell2D(Cell2D):
    """Defines the reciprocal cell for a crystal based on the lattice constants of the primitive cell."""
    lattice_constant: LatticeConstant2D
    
    def __post_init__(self):
        """Initializes and builds the basis for the reciprocal cell."""
        self.basis = self.build()
        self.volume = self.calculate_volume()
    
    def build(self) -> np.ndarray:
        """Constructs the basis matrix for the reciprocal cell from the primitive cell.

        Returns:
            np.ndarray: The basis matrix of the reciprocal cell.
        """
        primitive_cell = PrimitiveCell2D(self.lattice_constant)

        basis = np.zeros((2,2))
        
        primitive_vector = primitive_cell.basis

        quarter_rotation_matrix = np.array([
            [0.0, -1.0],
            [1.0, 0.0]
        ])
        
        for i in range(2):
            j = (i+1) % 2
            rotated_primitive_vector = quarter_rotation_matrix @ primitive_vector[j]
            basis[i] = rotated_primitive_vector / np.dot(primitive_vector[i], rotated_primitive_vector) * 2.0 * np.pi
        
        return basis

    def get_reciprocal_vectors(self, n_g: int) -> np.ndarray:
        """Generate the reciprocal lattice vectors used to define the Brillouin zone boundaries.

        Returns:
            np.ndarray: An array of reciprocal lattice vectors.
        """

        n_cut = np.ceil(np.sqrt(n_g))

        n_1d = np.arange(-n_cut, n_cut + 1)
        n_2d = np.array(np.meshgrid(n_1d, n_1d)).T.reshape(-1, 2)
        
        g = n_2d @ self.basis
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

    def get_monkhorst_pack_grid(self, n_x: int, n_y: int) -> np.ndarray:
        x = (2 * np.arange(1, n_x + 1) - n_x - 1) / (2 * n_x)
        y = (2 * np.arange(1, n_y + 1) - n_y - 1) / (2 * n_y)

        bx = np.broadcast_to(x[:, np.newaxis], (n_x, n_y))
        by = np.broadcast_to(y[np.newaxis, :], (n_x, n_y))

        aligned_k = np.stack([bx, by], axis=-1).reshape(-1, 2) @ self.basis

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
        k = np.empty((n_via * n_split, 2))

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
        if self.lattice_constant.crystal_structure == 'square':
            special_point = SpecialPoints2D.Square[k_name]
        elif self.lattice_constant.crystal_structure == 'hexagonal':
            special_point = SpecialPoints2D.Hexagonal[k_name]
        else:
            raise ValueError("Invalid name specified.")
        
        return np.array(special_point) @ self.basis

@dataclass
class ReciprocalCell1D(Cell1D):
    """Defines the reciprocal cell for a crystal based on the lattice constants of the primitive cell."""
    lattice_constant: LatticeConstant1D
    
    def __post_init__(self):
        """Initializes and builds the basis for the reciprocal cell."""
        self.basis = self.build()
        self.volume = self.calculate_volume()
    
    def build(self) -> np.ndarray:
        """Constructs the basis matrix for the reciprocal cell from the primitive cell.

        Returns:
            np.ndarray: The basis matrix of the reciprocal cell.
        """
        primitive_cell = PrimitiveCell1D(self.lattice_constant)
        
        primitive_vector = primitive_cell.basis

        basis = 2.0 * np.pi / primitive_vector
        
        return basis

    def get_reciprocal_vectors(self, n_g: int) -> np.ndarray:
        """Generate the reciprocal lattice vectors used to define the Brillouin zone boundaries.

        Returns:
            np.ndarray: An array of reciprocal lattice vectors.
        """

        n_cut = np.ceil(np.abs(n_g))

        n_1d = np.arange(-n_cut, n_cut + 1)
        
        g = n_1d * self.basis
        g_norm = abs(g).round(decimals=5)
        g_norm_unique = np.unique(g_norm)

        g_list = []

        for g_ref in g_norm_unique:
            count = 0
            for g_compare in g_norm:
                if g_compare == g_ref:
                    g_list.append(g[count])
                count += 1

        return np.array(g_list[0:n_g])

    def get_monkhorst_pack_grid(self, n_x: int) -> np.ndarray:
        x = (2 * np.arange(1, n_x + 1) - n_x - 1) / (2 * n_x)

        aligned_k = x * self.basis

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
        k = np.empty((n_via * n_split,))

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
        special_point = SpecialPoints1D.Line[k_name]
        
        return special_point * self.basis