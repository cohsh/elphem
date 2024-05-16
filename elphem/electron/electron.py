import math
import numpy as np

from elphem.lattice.lattice import Lattice
from elphem.lattice.path import PathValues

class Electron:
    """Represents a free electron in a given empty lattice.
    
    Attributes:
        lattice (Lattice): The lattice on which the free electron model is applied.
        n_electrons (int): Number of electrons per unit cell.
        n_bands (int): Number of energy bands considered.
        n_k (int): Number of k_points.
        g (np.ndarray): Reciprocal lattice vectors.
        k (np.ndarray): Vectors in the reciprocal space.
        eigenenergies (np.ndarray): Eigenenergies of free electron model.
        fermi_energy (float): Fermi energy
    """
    
    def __init__(self, lattice: Lattice, n_electrons: int):
        self.lattice = lattice
        self.n_electrons = n_electrons
        self.n_bands = None
        self.n_k = None
        self.g = None
        self.k = None
        self.eigenenergies = None

        self.fermi_energy = self.calculate_fermi_energy()
        
    @classmethod
    def create_from_n(cls, lattice: Lattice, n_electrons: int, n_bands: int, n_k_array: np.ndarray) -> 'Electron':
        electron = Electron(lattice, n_electrons)

        electron.k = lattice.reciprocal.get_monkhorst_pack_grid(*n_k_array)
        electron.n_k = np.prod(n_k_array)

        electron.update_band(n_bands)
        electron.update_eigenenergies()
        
        return electron

    @classmethod
    def create_from_k(cls, lattice: Lattice, n_electrons: int, n_bands: int, k_array: np.ndarray) -> 'Electron':
        electron = Electron(lattice, n_electrons)

        if isinstance(k_array, list):
            k_array = np.array(k_array)

        if k_array.shape == (lattice.n_dim,):
            k_array = np.array([k_array])

        electron.n_k = len(k_array)
        electron.k = k_array
        
        electron.update_band(n_bands)
        electron.update_eigenenergies()
        
        return electron

    @classmethod
    def create_from_gk_grid(cls, lattice: Lattice, n_electrons: int, g_array: np.ndarray, k_array: np.ndarray) -> 'Electron':
        electron = Electron(lattice, n_electrons)
        
        electron.g = g_array
        electron.k = k_array
        
        electron.update_eigenenergies(expand_g=False)
        
        return electron
    
    @classmethod
    def create_from_path(cls, lattice: Lattice, n_electrons: int, n_bands: int, k_path: PathValues) -> 'Electron':
        electron = Electron.create_from_k(lattice, n_electrons, n_bands, k_path.values)
        
        return electron

    def clone_with_gk_grid(self, g_array: np.ndarray, k_array: np.ndarray) -> 'Electron':
        electron = self.create_from_gk_grid(self.lattice, self.n_electrons, g_array, k_array)
        electron.n_k = self.n_k
        electron.n_bands = self.n_bands
        
        return electron

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
            eigenenergies = np.array([0.5 * np.linalg.norm(k_array + g, axis=-1) ** 2 for g in g_array])
            eigenenergies -= self.fermi_energy
        
        return eigenenergies

    def calculate_dos(self, omega: float | np.ndarray) -> np.ndarray:
        omega_plus_fermi_energy = omega + self.fermi_energy
        if self.lattice.n_dim == 3:
            coefficient = 8.0 * np.pi / self.lattice.reciprocal.volume
            return coefficient * np.sqrt(2.0 * omega_plus_fermi_energy)
        elif self.lattice.n_dim == 2:
            coefficient = 4.0 * np.pi / self.lattice.reciprocal.volume
            return coefficient
        else:
            coefficient = 2.0 / self.lattice.reciprocal.volume
            return coefficient / np.sqrt(2.0 * omega_plus_fermi_energy)

    def calculate_eigenenergies_with_path(self, k_path: PathValues) -> PathValues:
        eigenenergies = self.calculate_eigenenergies(k_path.values, self.g)
        
        return k_path.derive(eigenenergies)

    def update_band(self, n_bands: int) -> None:
        self.n_bands = n_bands
        self.g = self.lattice.reciprocal.get_reciprocal_vectors(self.n_bands)

    def update_eigenenergies(self, expand_g: bool = True) -> None:
        if expand_g:
            self.eigenenergies = self.calculate_eigenenergies(self.k, self.g)
        else:
            self.eigenenergies = self.calculate_eigenenergies(self.k + self.g)

    def calculate_fermi_energy(self) -> float:
        """Calculate the Fermi energy.

        Returns:
            float: The Fermi energy.
        """
        gamma = math.gamma(self.lattice.n_dim / 2.0 + 1.0)
        coefficient = 2.0 * np.pi
        electron_density = self.n_electrons / self.lattice.primitive.volume
        
        fermi_energy = coefficient * (0.5 * gamma * electron_density) ** (2.0 / self.lattice.n_dim)
        
        return fermi_energy