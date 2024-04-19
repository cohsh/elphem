from dataclasses import dataclass
import numpy as np

from elphem.const.atomic_weight import AtomicWeight
from elphem.const.unit import Mass
from elphem.lattice.cell import PrimitiveCell, ReciprocalCell
from elphem.lattice.lattice_constant import LatticeConstant
from elphem.lattice.grid import MonkhorstPackGrid


@dataclass
class Lattice:
    """A class to simulate an empty lattice for a given crystal structure and lattice constant a."""
    crystal_structure: str
    atoms: str | list[str] | np.ndarray
    a: float

    def __post_init__(self):
        """Initializes the lattice constants, primitive, and reciprocal cells along with their volumes and bases."""
        self.set_constants()
        self.correct_atoms()
        self.set_about_atoms()

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
    
    def get_grid(self, n_x: int, n_y: int, n_z: int) -> MonkhorstPackGrid:
        grid = MonkhorstPackGrid(self.basis["reciprocal"], n_x, n_y, n_z)
        
        return grid

    def get_reciprocal_vectors(self, n_g: int) -> np.ndarray:
        """Generate the reciprocal lattice vectors used to define the Brillouin zone boundaries.

        Returns:
            np.ndarray: An array of reciprocal lattice vectors.
        """

        n_cut = np.ceil(np.cbrt(n_g))

        n_1d = np.arange(-n_cut, n_cut + 1)
        n_3d = np.array(np.meshgrid(n_1d, n_1d, n_1d)).T.reshape(-1, 3)
        
        g = n_3d @ self.basis["reciprocal"]
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

    def set_constants(self) -> None:
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
            self.constants = LatticeConstant(self.a, self.a, self.a, alpha, alpha, alpha, crystal_structure_lower)
        else:
            raise ValueError("Invalid crystal structure specified.")
    
    def correct_atoms(self) -> None:
        if isinstance(self.atoms, str):
            self.atoms = [self.atoms]
        elif isinstance(self.atoms, np.ndarray):
            self.atoms = self.atoms.tolist()

    def set_about_atoms(self) -> None:
        self.n_atoms = len(self.atoms)

        masses = np.array(AtomicWeight.get_from_list(self.atoms)) * Mass.DALTON["->"]
        self.mass = np.average(masses)