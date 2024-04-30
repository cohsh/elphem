import numpy as np

from elphem.common.unit import Energy
from elphem.common.function import safe_divide
from elphem.common.distribution import bose_distribution
from elphem.lattice.lattice import Lattice
from elphem.lattice.path import PathValues

class DebyePhonon:
    """Models the phononic properties of a lattice using the Debye model.

    Attributes:
        lattice (EmptyLattice): The crystal lattice on which the model is applied.
        debye_temperature (float): The Debye temperature of the lattice material.
        number_of_atom (float): The number of atoms per primitive cell.
        mass (float): The mass of the crystal's atoms.
    """

    def __init__(self, lattice: Lattice, debye_temperature: float) -> None:
        """Validate initial model parameters."""
        self.debye_temperature = debye_temperature
        self.n_q = np.prod(self.n_q_array)
        self.temperature = lattice.temperature
        self.speed_of_sound = self.calculate_speed_of_sound(lattice)
        
        self.q = None
        self.eigenenergies = None
        self.eigenvectors = None
        self.zero_point_lengths = None
        self.occupations = None
        
    def create_from_k(self, lattice: Lattice, debye_temperature: float, q_array: np.ndarray) -> 'DebyePhonon':
        debye_phonon = DebyePhonon(lattice, debye_temperature)
    
        debye_phonon.n_q = len(q_array)
        debye_phonon.q = q_array
        
        debye_phonon.eigenenergies = debye_phonon.calculate_eigenenergies(q_array)
        debye_phonon.eigenvectors = debye_phonon.calculate_eigenvectors(q_array)
        debye_phonon.zero_point_lengths = safe_divide(1.0, np.sqrt(2.0 * lattice.mass * debye_phonon.eigenenergies))
        debye_phonon.occupations = bose_distribution(debye_phonon.temperature, debye_phonon.eigenenergies)
        
        return debye_phonon
    
    def create_from_n(self, lattice: Lattice, debye_temperature: float, n_q_array: np.ndarray | list[int]) -> 'DebyePhonon':
        debye_phonon = DebyePhonon(lattice, debye_temperature)

        debye_phonon.n_q = np.prod(self.n_q_array)
        debye_phonon.q = lattice.reciprocal.get_monkhorst_pack_grid(*n_q_array)

        debye_phonon.eigenenergies = debye_phonon.calculate_eigenenergies(debye_phonon.q)
        debye_phonon.eigenvectors = debye_phonon.calculate_eigenvectors(debye_phonon.q)
        debye_phonon.zero_point_lengths = safe_divide(1.0, np.sqrt(2.0 * lattice.mass * debye_phonon.eigenenergies))
        debye_phonon.occupations = bose_distribution(debye_phonon.temperature, debye_phonon.eigenenergies)
        
        return debye_phonon


    def calculate_eigenenergies(self, q_array: np.ndarray = None) -> np.ndarray:
        """Calculate phonon eigenenergies at wave vector q.

        Args:
            q (np.ndarray): A numpy array representing vectors in reciprocal space.

        Returns:
            np.ndarray: The phonon eigenenergies at each wave vector.
        """
        if q_array is None:
            eigenenergies = self.speed_of_sound * np.linalg.norm(self.q, axis=-1)
        else:
            eigenenergies = self.speed_of_sound * np.linalg.norm(q_array, axis=-1)
        
        return eigenenergies

    def get_eigenenergies_with_path(self, q_names: list[np.ndarray], n_split) -> PathValues:
        """Calculate the phonon dispersion curves along specified paths in reciprocal space.

        Args:
            q_names (list[np.ndarray]): List of special points defining the path through the Brillouin zone.
            n_split (int): Number of points between each special point to compute the dispersion curve.

        Returns:
            tuple: A tuple containing the x-coordinates for plotting, omega (eigenenergy values), and x-coordinates of special points.
        """

        q_path = self.lattice.reciprocal.get_path(q_names, n_split)

        eigenenergies = self.get_eigenenergies(q_path.values)
        
        return q_path.derive(eigenenergies)

    def calculate_eigenvectors(self, q_array: np.ndarray = None) -> np.ndarray:
        """Calculate phonon eigenvectors at wave vector q.

        Args:
            q (np.ndarray): A numpy array representing vectors in reciprocal space.

        Returns:
            np.ndarray: The phonon eigenvectors at each wave vector, represented as complex numbers.
        """

        if q_array is None:
            q_norm = np.repeat(np.linalg.norm(self.q, axis=-1, keepdims=True), self.q.shape[-1], axis=-1)
            eigenvectors = 1.0j * safe_divide(self.q, q_norm)
        else:
            q_norm = np.repeat(np.linalg.norm(q_array, axis=-1, keepdims=True), q_array.shape[-1], axis=-1)
            eigenvectors = 1.0j * safe_divide(q_array, q_norm)

        return eigenvectors

    def update(self, q: np.ndarray) -> None:
        self.q = q
        self.n_q = len(q)
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

        speed_of_sound = debye_frequency * (6.0 * np.pi ** 2 * number_density) ** (-1.0/3.0)
        
        return speed_of_sound