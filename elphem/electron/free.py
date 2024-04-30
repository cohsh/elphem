import numpy as np

from elphem.common.distribution import fermi_distribution
from elphem.lattice.lattice import Lattice
from elphem.lattice.path import PathValues

class FreeElectron:
    """Represents a free electron model on a given crystal lattice.
    
    Attributes:
        lattice (EmptyLattice): The lattice on which the free electron model is applied.
        n_band (int): Number of energy bands considered.
        n_electron (int): Number of electrons per unit cell.
    """
    
    def __init__(self, lattice: Lattice, n_band: int, n_electron: int):
        self.g = lattice.reciprocal.get_reciprocal_vectors(self.n_band)
        self.n_band = n_band
        self.n_electron = n_electron
        self.temperature = lattice.temperature
        self.fermi_energy = self.calculate_fermi_energy(lattice)
        
        self.k = None
        self.n_k = None
        self.eigenenergies = None
        self.occupations = None

    def create_from_k(self, lattice: Lattice, n_band: int, n_electron: int, k_array: np.ndarray) -> 'FreeElectron':
        free_electron = FreeElectron(lattice, n_band, n_electron)

        free_electron.n_k = len(k_array)
        free_electron.k = k_array
        
        free_electron.eigenenergies = free_electron.calculate_eigenenergies(k_array, free_electron.g)
        free_electron.occupations = fermi_distribution(free_electron.temperature, free_electron.eigenenergies)
        
        return free_electron
        
    def create_from_n(self, lattice: Lattice, n_band: int, n_electron: int, n_k_array: np.ndarray) -> 'FreeElectron':
        free_electron = FreeElectron(lattice, n_band, n_electron)

        free_electron.n_k = np.prod(n_k_array)
        free_electron.k = lattice.reciprocal.get_monkhorst_pack_grid(*n_k_array)

        free_electron.eigenenergies = free_electron.calculate_eigenenergies(free_electron.k, free_electron.g)
        free_electron.occupations = fermi_distribution(free_electron.temperature, free_electron.eigenenergies)
        
        return free_electron

    def calculate_eigenenergies(self, k_array: np.ndarray, g_array: np.ndarray = None) -> np.ndarray:
        """Calculate the electron eigenenergies at wave vector k.
        
        Args:
            k_array (Optional[np.ndarray]): A numpy array representing vectors in reciprocal space.
            g_array (Optional[np.ndarray]): An optional numpy array of vectors to be added to `k_array`.

        Returns:
            np.ndarray: The electron eigenenergies at each wave vector.
        """
        if g_array is None:
            eigenenergies =  0.5 * np.linalg.norm(k_array, axis=-1) ** 2 - self.fermi_energy
        else:
            eigenenergies = np.array([0.5 * np.linalg.norm(k_array + g, axis=-1) ** 2 - self.fermi_energy for g in g_array])
        
        return eigenenergies

    def update(self, k: np.ndarray, g: np.ndarray = None) -> None:
        self.k = k
        self.n_k = len(k)
        
        if g is not None:
            self.eigenenergies = self.get_eigenenergies(k + g)
            self.g = g
            self.occupations = fermi_distribution(self.temperature, self.eigenenergies)
        else:
            self.eigenenergies = self.get_eigenenergies(k, self.g)
            self.occupations = fermi_distribution(self.temperature, self.eigenenergies)
        
    def calculate_fermi_energy(self, lattice: Lattice) -> float:
        """Calculate the Fermi energy of the electron system.

        Returns:
            float: The Fermi energy.
        """
        electron_density = self.n_electron / lattice.primitive.volume
        fermi_energy = 0.5 * (3 * np.pi ** 2 * electron_density) ** (2/3)
        
        return fermi_energy