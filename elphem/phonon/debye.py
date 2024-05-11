import numpy as np

from elphem.common.unit import Energy
from elphem.common.function import safe_divide
from elphem.common.distribution import bose_distribution
from elphem.lattice.lattice import Lattice3D, Lattice2D, Lattice1D
from elphem.lattice.path import PathValues

class DebyePhonon:
    """Models the phononic properties of a lattice using the Debye model.

    Attributes:
        lattice (EmptyLattice): The crystal lattice on which the model is applied.
        debye_temperature (float): The Debye temperature of the lattice material.
        number_of_atom (float): The number of atoms per primitive cell.
        mass (float): The mass of the crystal's atoms.
    """

    def __init__(self, lattice: Lattice3D | Lattice2D | Lattice1D, debye_temperature: float) -> None:
        """Validate initial model parameters."""
        self.lattice = lattice
        self.debye_temperature = debye_temperature
        self.temperature = lattice.temperature
        self.speed_of_sound = self.calculate_speed_of_sound(lattice)
        
        self.q = None
        self.n_q = None
        self.eigenenergies = None
        self.eigenvectors = None
        self.zero_point_lengths = None
        self.occupations = None
    
    @classmethod
    def create_from_q(cls, lattice: Lattice3D | Lattice2D | Lattice1D, debye_temperature: float, q_array: np.ndarray) -> 'DebyePhonon':
        debye_phonon = DebyePhonon(lattice, debye_temperature)
    
        debye_phonon.n_q = len(q_array)
        debye_phonon.q = q_array
        
        debye_phonon.eigenenergies = debye_phonon.calculate_eigenenergies(debye_phonon.q)
        debye_phonon.eigenvectors = debye_phonon.calculate_eigenvectors(debye_phonon.q)
        debye_phonon.zero_point_lengths = safe_divide(1.0, np.sqrt(2.0 * lattice.mass * debye_phonon.eigenenergies))
        debye_phonon.occupations = bose_distribution(debye_phonon.temperature, debye_phonon.eigenenergies)
        
        return debye_phonon
    
    @classmethod
    def create_from_n(cls, lattice: Lattice3D | Lattice2D | Lattice1D, debye_temperature: float, n_q_array: np.ndarray | list[int]) -> 'DebyePhonon':
        debye_phonon = DebyePhonon(lattice, debye_temperature)

        debye_phonon.n_q = np.prod(n_q_array)
        debye_phonon.q = lattice.reciprocal.get_monkhorst_pack_grid(*n_q_array)

        debye_phonon.eigenenergies = debye_phonon.calculate_eigenenergies(debye_phonon.q)
        debye_phonon.eigenvectors = debye_phonon.calculate_eigenvectors(debye_phonon.q)
        debye_phonon.zero_point_lengths = safe_divide(1.0, np.sqrt(2.0 * lattice.mass * debye_phonon.eigenenergies))
        debye_phonon.occupations = bose_distribution(debye_phonon.temperature, debye_phonon.eigenenergies)
        
        return debye_phonon
    
    @classmethod
    def create_from_path(cls, lattice: Lattice3D | Lattice2D | Lattice1D, debye_temperature: float, q_path: PathValues) -> 'DebyePhonon':
        debye_phonon = DebyePhonon.create_from_q(lattice, debye_temperature, q_path.values)
        
        return debye_phonon

    def clone_with_q_grid(self, q_array: np.ndarray) -> 'DebyePhonon':
        debye_phonon = self.create_from_q(self.lattice, self.debye_temperature, q_array)
        debye_phonon.n_q = self.n_q
        
        return debye_phonon

    def calculate_eigenenergies(self, q_array: np.ndarray) -> np.ndarray:
        """Calculate phonon eigenenergies at wave vector q.

        Args:
            q (np.ndarray): A numpy array representing vectors in reciprocal space.

        Returns:
            np.ndarray: The phonon eigenenergies at each wave vector.
        """
        eigenenergies = self.speed_of_sound * np.linalg.norm(q_array, axis=-1)
        
        return eigenenergies

    def calculate_eigenvectors(self, q_array: np.ndarray) -> np.ndarray:
        """Calculate phonon eigenvectors at wave vector q.

        Args:
            q (np.ndarray): A numpy array representing vectors in reciprocal space.

        Returns:
            np.ndarray: The phonon eigenvectors at each wave vector, represented as complex numbers.
        """
        q_norm = np.repeat(np.linalg.norm(q_array, axis=-1, keepdims=True), q_array.shape[-1], axis=-1)
            
        eigenvectors = 1.0j * safe_divide(q_array, q_norm)

        return eigenvectors

    def update(self, q: np.ndarray) -> None:
        self.q = q
        self.eigenenergies = self.get_eigenenergies(q)
        self.eigenvectors = self.get_eigenvectors(q)
        self.zero_point_lengths = safe_divide(1.0, np.sqrt(2.0 * self.lattice.mass * self.eigenenergies))
        self.occupations = bose_distribution(self.temperature, self.eigenenergies)

    def calculate_speed_of_sound(self, lattice) -> float:
        """Calculate the speed of sound in the lattice based on Debye model.

        Returns:
            float: The speed of sound in Hartree atomic units.
        """
        try:
            number_density = lattice.n_atoms / lattice.primitive.volume
        except ZeroDivisionError:
            ValueError("Lattice volume must be positive.")


        debye_frequency = self.debye_temperature * Energy.KELVIN["->"]

        if lattice.n_dim == 3:
            speed_of_sound = debye_frequency * (6.0 * np.pi ** 2 * number_density) ** (-1.0/3.0)
        elif lattice.n_dim == 2:
            speed_of_sound = debye_frequency * (4.0 * np.pi * number_density) ** (-1.0/2.0)
        elif lattice.n_dim == 1:
            speed_of_sound = debye_frequency / (2.0 * np.pi * number_density)
        
        return speed_of_sound