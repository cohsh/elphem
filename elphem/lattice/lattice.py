from dataclasses import dataclass
import numpy as np

from elphem.common.atomic_weight import AtomicWeight
from elphem.common.unit import Mass
from elphem.lattice.cell import PrimitiveCell, ReciprocalCell
from elphem.lattice.lattice_constant import LatticeConstant

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
    
    def set_about_atoms(self) -> None:
        self.n_atoms = len(self.atoms)

        masses = np.array(AtomicWeight.get_from_list(self.atoms)) * Mass.DALTON["->"]
        self.mass = np.average(masses)

    def correct_atoms(self) -> None:
        if isinstance(self.atoms, str):
            self.atoms = [self.atoms]
        elif isinstance(self.atoms, np.ndarray):
            self.atoms = self.atoms.tolist()