from dataclasses import dataclass
import numpy as np

from elphem.common.atomic_weight import AtomicWeight
from elphem.common.unit import Mass
from elphem.lattice.primitive_cell import PrimitiveCell3D, PrimitiveCell2D, PrimitiveCell1D
from elphem.lattice.reciprocal_cell import ReciprocalCell3D, ReciprocalCell2D, ReciprocalCell1D
from elphem.lattice.lattice_constant import LatticeConstant3D, LatticeConstant2D, LatticeConstant1D
from elphem.lattice.path import PathValues

@dataclass
class Lattice:
    """A class to simulate an empty lattice for a given crystal structure and lattice constant a.
    Attributes:
        crystal_structure (str): A string for crystal structures
        atoms (str | list[str] | np.ndarray): Name(s) of atoms
        n_atoms (int): Number of atoms
        a (float): A lattice constant
        primitive (PrimitiveCell): A primitive cell
        reciprocal (ReciprocalCell): A reciprocal cell
        mass (float): Mass of the empty lattice
    """
    crystal_structure: str
    atoms: str | list[str] | np.ndarray
    a: float
    primitive = None
    reciprocal = None

    def __post_init__(self):
        self.correct_atoms()
        self.set_about_atoms()

    def set_about_atoms(self) -> None:
        """Set number of atoms and mass.
        """
        self.n_atoms = len(self.atoms)

        masses = np.array(AtomicWeight.get_from_list(self.atoms)) * Mass.DALTON["->"]
        self.mass = np.average(masses)

    def correct_atoms(self) -> None:
        """Correct the type of atoms
        """
        if isinstance(self.atoms, str):
            self.atoms = [self.atoms]
        elif isinstance(self.atoms, np.ndarray):
            self.atoms = self.atoms.tolist()
    
    def get_k_path(self, k_names: list[str], n_split: int) -> PathValues:
        """Get a path in the reciprocal space

        Args:
            k_names (list[str]): A list for names of special k points.
            n_split (int): Number of points between each special point.

        Returns:
            PathValues: Values with path
        """
        return self.reciprocal.get_path(k_names, n_split)

class Lattice3D(Lattice):
    """A class for 3D lattice"""
    def __init__(self, crystal_structure: str, atoms: str | list[str] | np.ndarray, a: float):
        super().__init__(crystal_structure, atoms, a)
        self.n_dim = 3
        self.set_constants()

        self.primitive = PrimitiveCell3D(self.constants)
        self.reciprocal = ReciprocalCell3D(self.constants)

    def set_constants(self) -> None:
        """Determines lattice constants based on the crystal structure.
        
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
            self.constants = LatticeConstant3D(self.a, self.a, self.a, alpha, alpha, alpha, crystal_structure_lower)
        else:
            raise ValueError("Invalid crystal structure specified.")

class Lattice2D(Lattice):
    """A class for 2D lattice"""
    def __init__(self, crystal_structure: str, atoms: str | list[str] | np.ndarray, a: float):
        super().__init__(crystal_structure, atoms, a)
        self.n_dim = 2
        self.set_constants()

        self.primitive = PrimitiveCell2D(self.constants)
        self.reciprocal = ReciprocalCell2D(self.constants)

    def set_constants(self) -> None:
        """Determines lattice constants based on the crystal structure.
        
        Raises:
            ValueError: If an invalid crystal structure name is specified.
        """
        crystal_structure_lower = self.crystal_structure.lower()

        alpha_values = {
            'square': 90.0,
            'hexagonal': 120.0
        }

        alpha = alpha_values.get(crystal_structure_lower)

        if alpha is not None:
            self.constants = LatticeConstant2D(self.a, self.a, alpha, crystal_structure_lower)
        else:
            raise ValueError("Invalid crystal structure specified.")

class Lattice1D(Lattice):
    """A class for 1D lattice"""
    def __init__(self, atoms: str | list[str] | np.ndarray, a: float):
        super().__init__('', atoms, a)
        self.n_dim = 1
        self.set_constants()

        self.primitive = PrimitiveCell1D(self.constants)
        self.reciprocal = ReciprocalCell1D(self.constants)

    def set_constants(self) -> None:
        """Determines lattice constants based on the crystal structure.
        
        Raises:
            ValueError: If an invalid crystal structure name is specified.
        """

        self.constants = LatticeConstant1D(self.a)