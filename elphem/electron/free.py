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
    
    def __init__(self, lattice: Lattice, n_electron: int):
        self.lattice = lattice
        self.n_electron = n_electron
        self.temperature = lattice.temperature
        self.fermi_energy = self.calculate_fermi_energy(lattice)

        self.n_band = None
        self.g = None
        self.k = None
        self.n_k = None
        self.eigenenergies = None
        self.occupations = None
        
    @classmethod
    def create_from_n(cls, lattice: Lattice, n_electron: int, n_band: int, n_k_array: np.ndarray) -> 'FreeElectron':
        free_electron = FreeElectron(lattice, n_electron)

        free_electron.k = lattice.reciprocal.get_monkhorst_pack_grid(*n_k_array)
        free_electron.n_k = np.prod(n_k_array)

        free_electron.update_band(lattice, n_band)
        free_electron.update_eigenenergies_and_occupations()
        
        return free_electron

    @classmethod
    def create_from_k(cls, lattice: Lattice, n_electron: int, n_band: int, k_array: np.ndarray) -> 'FreeElectron':
        free_electron = FreeElectron(lattice, n_electron)

        free_electron.k = k_array
        free_electron.n_k = len(k_array)

        free_electron.update_band(lattice, n_band)
        free_electron.update_eigenenergies_and_occupations()
        
        return free_electron

    @classmethod
    def create_from_gk_grid(cls, lattice: Lattice, n_electron: int, g_array: np.ndarray, k_array: np.ndarray) -> 'FreeElectron':
        free_electron = FreeElectron(lattice, n_electron)
        
        free_electron.g = g_array
        free_electron.k = k_array
        
        free_electron.update_eigenenergies_and_occupations(expand_g=False)
        
        return free_electron
    
    @classmethod
    def create_from_path(cls, lattice: Lattice, n_electron: int, n_band: int, k_path: PathValues) -> 'FreeElectron':
        free_electron = FreeElectron.create_from_k(lattice, n_electron, n_band, k_path.values)
        
        return free_electron

    def clone_with_gk_grid(self, g_array: np.ndarray, k_array: np.ndarray) -> 'FreeElectron':
        free_electron = self.create_from_gk_grid(self.lattice, self.n_electron, g_array, k_array)
        free_electron.n_k = self.n_k
        
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

    def calculate_occupations(self, eigenenergies: np.ndarray) -> np.ndarray:
        return fermi_distribution(self.temperature, eigenenergies)
    
    def calculate_eigenenergies_with_path(self, k_path: PathValues) -> PathValues:
        eigenenergies = self.calculate_eigenenergies(k_path.values, self.g)
        
        return k_path.derive(eigenenergies)

    def update_band(self, lattice: Lattice, n_band: int) -> None:
        self.n_band = n_band
        self.g = lattice.reciprocal.get_reciprocal_vectors(self.n_band)

    def update_eigenenergies_and_occupations(self, expand_g: bool = True) -> None:
        if expand_g:
            self.eigenenergies = self.calculate_eigenenergies(self.k, self.g)
        else:
            self.eigenenergies = self.calculate_eigenenergies(self.k + self.g)

        self.occupations = self.calculate_occupations(self.eigenenergies)

    def update(self, g: np.ndarray, k: np.ndarray, expand_g: bool = False) -> None:
        self.k = k
        self.g = g
        
        self.update_eigenenergies_and_occupations(expand_g)
        
    def calculate_fermi_energy(self, lattice: Lattice) -> float:
        """Calculate the Fermi energy of the electron system.

        Returns:
            float: The Fermi energy.
        """
        electron_density = self.n_electron / lattice.primitive.volume
        fermi_energy = 0.5 * (3 * np.pi ** 2 * electron_density) ** (2/3)
        
        return fermi_energy