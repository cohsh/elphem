import math
import numpy as np

from elphem.lattice.lattice import Lattice
from elphem.lattice.path import PathValues

class Electron:
    """Represents free electrons in a given empty lattice.
    
    Attributes:
        lattice (Lattice): Lattice on which the free electron model is applied.
        n_electrons (int): Number of electrons per unit cell.
        n_bands (int): Number of energy bands considered.
        n_k (int): Number of k_points.
        g (np.ndarray): Reciprocal lattice vectors.
        k (np.ndarray): Vectors in the reciprocal space.
        eigenenergies (np.ndarray): Eigenenergies of free electron model.
        fermi_energy (float): Fermi energy
        fermi_wave_number (float): Fermi wave-number
        thomas_fermi_wave_number (float): Thomas-Fermi wave-number
    """
    
    def __init__(self, lattice: Lattice, n_electrons: int):
        """
        Args:
            lattice (Lattice): Lattice on which the free electron model is applied.
            n_electrons (int): Number of electrons per unit cell.
        """
        
        self.lattice = lattice
        self.n_electrons = n_electrons
        self.n_bands = None
        self.n_k = None
        self.g = None
        self.k = None
        self.eigenenergies = None

        self.fermi_energy = self.calculate_fermi_energy()
        self.fermi_wave_number = self.calculate_fermi_wave_number()
        self.thomas_fermi_wave_number = self.calculate_thomas_fermi_wave_number()
        
    @classmethod
    def create_from_n(cls, lattice: Lattice, n_electrons: int, n_bands: int, n_k_array: np.ndarray) -> 'Electron':
        """Create free electrons from the number of the number of k (array-type).

        Args:
            lattice (Lattice): Lattice on which the free electron model is applied
            n_electrons (int): Number of electrons
            n_bands (int): Number of electron bands
            n_k_array (np.ndarray): Number of k vectors (array-type)

        Returns:
            Electron: Free electron
        """
        electron = Electron(lattice, n_electrons)

        # set about k vectors
        electron.k = lattice.reciprocal.get_monkhorst_pack_grid(*n_k_array)
        electron.n_k = np.prod(n_k_array)

        # update values
        electron.update_band(n_bands)
        electron.update_eigenenergies()
        
        return electron

    @classmethod
    def create_from_k(cls, lattice: Lattice, n_electrons: int, n_bands: int, k_array: np.ndarray) -> 'Electron':
        """Create free electrons from k vectors.

        Args:
            lattice (Lattice): Lattice on which the free electron model is applied
            n_electrons (int): Number of electrons
            n_bands (int): Number of electron bands
            k_array (np.ndarray): k vectors

        Returns:
            Electron: Free electron
        """
        electron = Electron(lattice, n_electrons)

        # check the type and shape fo k_array
        if isinstance(k_array, list):
            k_array = np.array(k_array)

        if k_array.shape == (lattice.n_dim,):
            k_array = np.array([k_array])

        # set about k vectors
        electron.n_k = len(k_array)
        electron.k = k_array

        # update values
        electron.update_band(n_bands)
        electron.update_eigenenergies()
        
        return electron

    @classmethod
    def create_from_gk_grid(cls, lattice: Lattice, n_electrons: int, g_array: np.ndarray, k_array: np.ndarray) -> 'Electron':
        """Create free electrons from G-k grid.

        Args:
            lattice (Lattice): Lattice on which the free electron model is applied
            n_electrons (int): Number of electrons
            g_array (np.ndarray): Reciprocal lattice vectors
            k_array (np.ndarray): k vectors

        Returns:
            Electron: Free electron
        """
        electron = Electron(lattice, n_electrons)
        
        electron.g = g_array
        electron.k = k_array
        
        electron.update_eigenenergies(expand_g=False)
        
        return electron
    
    @classmethod
    def create_from_path(cls, lattice: Lattice, n_electrons: int, n_bands: int, k_path: PathValues) -> 'Electron':
        """Create free electrons from k-path.

        Args:
            lattice (Lattice): Lattice on which the free electron model is applied
            n_electrons (int): Number of electrons
            n_bands (int): Number of bands
            k_path (PathValues): Path in the reciprocal lattice

        Returns:
            Electron: Free electron
        """
        electron = Electron.create_from_k(lattice, n_electrons, n_bands, k_path.values)
        
        return electron

    def clone_with_gk_grid(self, g_array: np.ndarray, k_array: np.ndarray) -> 'Electron':
        """Clone a free electron with changing G and k vectors.

        Args:
            g_array (np.ndarray): Reciprocal lattice vectors
            k_array (np.ndarray): k vectors

        Returns:
            Electron: Free electron
        """
        electron = self.create_from_gk_grid(self.lattice, self.n_electrons, g_array, k_array)

        # set about numbers
        electron.n_k = self.n_k
        electron.n_bands = self.n_bands
        
        return electron

    def calculate_eigenenergies(self, k_array: np.ndarray, g_array: np.ndarray = None) -> np.ndarray:
        """Calculate the electron eigenenergies at wave vector k.
        
        Args:
            k_array (np.ndarray): k vectors
            g_array (np.ndarray): Reciprocal lattice vectors

        Returns:
            np.ndarray: The electron eigenenergies
        """
        if g_array is None:
            eigenenergies =  0.5 * np.linalg.norm(k_array, axis=-1) ** 2 - self.fermi_energy
        else:
            eigenenergies = np.array([0.5 * np.linalg.norm(k_array + g, axis=-1) ** 2 for g in g_array])
            eigenenergies -= self.fermi_energy
        
        return eigenenergies

    def calculate_dos(self, omega: float | np.ndarray) -> np.ndarray:
        """Calculate DOS by using analytical forms.

        Args:
            omega (float | np.ndarray): frequencies

        Returns:
            np.ndarray: DOS
        """
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

    def update_band(self, n_bands: int) -> None:
        """Update attributes about bands

        Args:
            n_bands (int): Number of bands
        """
        self.n_bands = n_bands
        self.g = self.lattice.reciprocal.get_reciprocal_vectors(self.n_bands)

    def update_eigenenergies(self, expand_g: bool = True) -> None:
        """Update eigenenergies

        Args:
            expand_g (bool, optional): add axis about reciprocal lattice vectors. Defaults to True.
        """
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
        
        return coefficient * (0.5 * gamma * electron_density) ** (2.0 / self.lattice.n_dim)
    
    def calculate_fermi_wave_number(self) -> float:
        """Calculate the Fermi wave-number from Fermi energy.

        Returns:
            float: The Fermi wave-number
        """
        
        return np.sqrt(2.0 * self.fermi_energy)
    
    def calculate_thomas_fermi_wave_number(self) -> float:
        """Calculate the Thomas-Fermi wave-number from Fermi wave-number

        Returns:
            float: The Thomas-Fermi wave-number
        """
        
        coefficient = 4.0 / np.pi * (4.0 / (9.0 * np.pi)) ** (1.0 / 3.0)
        r = (self.lattice.primitive.volume * math.gamma(self.lattice.n_dim / 2.0 + 1.0) / (self.n_electrons * np.pi ** (self.lattice.n_dim / 2.0))) ** (1.0 / self.lattice.n_dim)
        
        return coefficient * r * self.fermi_wave_number ** 2