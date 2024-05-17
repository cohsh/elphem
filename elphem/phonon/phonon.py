import numpy as np

from elphem.common.unit import Energy
from elphem.common.function import safe_divide
from elphem.lattice.lattice import Lattice
from elphem.lattice.path import PathValues

class Phonon:
    """ Phonon with Debye model.

    Attributes:
        lattice (Lattice): A given empty lattice.
        debye_temperature (float): The Debye temperature for phonons.
        q (np.ndarray): Vectors in a given reciprocal space.
        n_q (int): Number of q points.
        eigenenergies (np.ndarray): Eigenenergies of phonon in Debye model.
        eigenvectors (np.ndarray): Eigenvectors of phonon in Debye model.
        zero_point_lengths (np.ndarray): Zero point lengths in Debye model.
        speed_of_sound (float): Speed of sound
    """

    def __init__(self, lattice: Lattice, debye_temperature: float) -> None:
        self.lattice = lattice
        self.debye_temperature = debye_temperature
        
        self.q = None
        self.n_q = None
        self.eigenenergies = None
        self.eigenvectors = None
        self.zero_point_lengths = None

        self.speed_of_sound = self.calculate_speed_of_sound()
    
    @classmethod
    def create_from_q(cls, lattice: Lattice, debye_temperature: float, q_array: np.ndarray) -> 'Phonon':
        """Create Debye phonon from q vectors

        Args:
            lattice (Lattice): Lattice on which the Debye model is applied
            debye_temperature (float): Debye temperature
            q_array (np.ndarray): q vectors

        Returns:
            Phonon: Debye phonon
        """
        phonon = Phonon(lattice, debye_temperature)

        # check the type and shape fo k_array
        if isinstance(q_array, list):
            q_array = np.array(q_array)

        if q_array.shape == (lattice.n_dim,):
            q_array = np.array([q_array])

        # set about q vectors
        phonon.q = q_array
        phonon.n_q = len(q_array)
        
        # set about values
        phonon.eigenenergies = phonon.calculate_eigenenergies(phonon.q)
        phonon.eigenvectors = phonon.calculate_eigenvectors(phonon.q)
        phonon.zero_point_lengths = phonon.calculate_zero_point_lengths()
        
        return phonon
    
    @classmethod
    def create_from_n(cls, lattice: Lattice, debye_temperature: float, n_q_array: np.ndarray | list[int]) -> 'Phonon':
        """Create Debye phonon from number of q vectors (array-type)

        Args:
            lattice (Lattice): Lattice on which the Debye model is applied
            debye_temperature (float): Debye temperature
            n_q_array (np.ndarray | list[int]): Number of q arrays

        Returns:
            Phonon: Debye phonon
        """
        phonon = Phonon(lattice, debye_temperature)
        
        # set about q vectors
        phonon.q = lattice.reciprocal.get_monkhorst_pack_grid(*n_q_array)
        phonon.n_q = np.prod(n_q_array)

        # set about values
        phonon.eigenenergies = phonon.calculate_eigenenergies(phonon.q)
        phonon.eigenvectors = phonon.calculate_eigenvectors(phonon.q)
        phonon.zero_point_lengths = phonon.calculate_zero_point_lengths()
        
        return phonon
    
    @classmethod
    def create_from_path(cls, lattice: Lattice, debye_temperature: float, q_path: PathValues) -> 'Phonon':
        """Create Debye phonon from q path

        Args:
            lattice (Lattice): Lattice on which the Debye model is applied
            debye_temperature (float): Debye model
            q_path (PathValues): q path

        Returns:
            Phonon: Debye phonon
        """
        phonon = Phonon.create_from_q(lattice, debye_temperature, q_path.values)
        
        return phonon

    def clone_with_q_grid(self, q_array: np.ndarray) -> 'Phonon':
        """Clone Debye phonon with changing q vectors

        Args:
            q_array (np.ndarray): q vectors

        Returns:
            Phonon: Debye phonon
        """
        phonon = self.create_from_q(self.lattice, self.debye_temperature, q_array)
        phonon.n_q = self.n_q
        
        return phonon

    def calculate_eigenenergies(self, q_array: np.ndarray) -> np.ndarray:
        """Calculate phonon eigenenergies.

        Args:
            q (np.ndarray): q vectors.

        Returns:
            np.ndarray: Phonon eigenenergies.
        """
        return self.speed_of_sound * np.linalg.norm(q_array, axis=-1)

    def calculate_eigenvectors(self, q_array: np.ndarray) -> np.ndarray:
        """Calculate phonon eigenvectors.

        Args:
            q (np.ndarray): q vectors

        Returns:
            np.ndarray: Phonon eigenvectors.
        """
        q_norm = np.repeat(np.linalg.norm(q_array, axis=-1, keepdims=True), q_array.shape[-1], axis=-1)
        
        return 1.0j * safe_divide(q_array, q_norm)

    def update(self, q: np.ndarray) -> None:
        """Update attributes

        Args:
            q (np.ndarray): q vectors
        """
        self.q = q
        self.eigenenergies = self.get_eigenenergies(q)
        self.eigenvectors = self.get_eigenvectors(q)
        self.zero_point_lengths = self.calculate_zero_point_lengths()

    def calculate_speed_of_sound(self) -> float:
        """Calculate the speed of sound.

        Returns:
            float: The speed of sound.
        """
        try:
            number_density = self.lattice.n_atoms / self.lattice.primitive.volume
        except ZeroDivisionError:
            ValueError("Lattice volume must be positive.")

        debye_frequency = self.debye_temperature * Energy.KELVIN["->"]

        if self.lattice.n_dim == 3:
            speed_of_sound = debye_frequency * (6.0 * np.pi ** 2 * number_density) ** (-1.0/3.0)
        elif self.lattice.n_dim == 2:
            speed_of_sound = debye_frequency * (4.0 * np.pi * number_density) ** (-1.0/2.0)
        elif self.lattice.n_dim == 1:
            speed_of_sound = debye_frequency / (2.0 * np.pi * number_density)
        
        return speed_of_sound
    
    def calculate_zero_point_lengths(self) -> np.ndarray:
        """Calculate zero point lengths

        Returns:
            np.ndarray: Zero point lengths
        """
        return safe_divide(1.0, np.sqrt(2.0 * self.lattice.mass * self.eigenenergies))