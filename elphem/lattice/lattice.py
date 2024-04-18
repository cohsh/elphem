from dataclasses import dataclass

from elphem.lattice.cell import PrimitiveCell, ReciprocalCell
from elphem.lattice.lattice_constant import LatticeConstant
from elphem.lattice.grid import MonkhorstPackGrid


@dataclass
class Lattice:
    """A class to simulate an empty lattice for a given crystal structure and lattice constant a."""
    crystal_structure: str
    a: float

    def __post_init__(self):
        """Initializes the lattice constants, primitive, and reciprocal cells along with their volumes and bases."""
        self.constants = self.set_constants()
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
    
    def get_grid(self, n_x, n_y, n_z) -> MonkhorstPackGrid:
        grid = MonkhorstPackGrid(self.basis["reciprocal"], n_x, n_y, n_z)
        
        return grid

    def set_constants(self) -> LatticeConstant:
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