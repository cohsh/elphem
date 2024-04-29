import numpy as np
from dataclasses import dataclass

from elphem.lattice.lattice import Lattice
from elphem.lattice.path import PathValues

@dataclass
class FreeElectron:
    """Represents a free electron model on a given crystal lattice.
    
    Attributes:
        lattice (EmptyLattice): The lattice on which the free electron model is applied.
        n_band (int): Number of energy bands considered.
        n_electron (int): Number of electrons per unit cell.
    """
    lattice: Lattice
    n_band: int
    n_electron: int
    n_k_array: np.ndarray | list[int]
    
    def __post_init__(self):
        self._set_fermi_energy()

        self.k = self.lattice.reciprocal_cell.get_monkhorst_pack_grid(*self.n_k_array)
        self.g = self.lattice.reciprocal_cell.get_reciprocal_vectors(self.n_band)

        self.eigenenergies = self.get_eigenenegies(self.k, self.g)

    def get_eigenenergies(self, k_array: np.ndarray = None, g_array: np.ndarray = None) -> np.ndarray:
        """Calculate the electron eigenenergies at wave vector k.
        
        Args:
            k_array (Optional[np.ndarray]): A numpy array representing vectors in reciprocal space.
            g_array (Optional[np.ndarray]): An optional numpy array of vectors to be added to `k_array`.

        Returns:
            np.ndarray: The electron eigenenergies at each wave vector.
        """
        if k_array is None and g_array is None:
            eigenenergies = np.array([0.5 * np.linalg.norm(self.k + g, axis=-1) ** 2 - self.fermi_energy for g in self.g])
        elif g_array is None:
            eigenenergies =  0.5 * np.linalg.norm(k_array, axis=-1) ** 2 - self.fermi_energy
        else:
            eigenenergies = np.array([0.5 * np.linalg.norm(k_array + g, axis=-1) ** 2 - self.fermi_energy for g in g_array])
        
        return eigenenergies

    def get_eigenenergies_with_path(self, k_names: list[str], n_split: int) -> PathValues:
        """Calculate the electronic band structures along the specified path in reciprocal space.

        Args:
            k_names (list[np.ndarray]): A list of special points names defining the path.
            n_split (int): Number of points between each special point to compute the band structure.

        Returns:
            tuple: A tuple containing x-coordinates for plotting, eigenenergy values, and x-coordinates of special points.
        """
        k_path = self.lattice.reciprocal.get_path(k_names, n_split)
        
        eigenenergies = self.get_eigenenergies(k_path.values, self.g)
        
        return k_path.derive(eigenenergies)
        
    def _set_fermi_energy(self) -> None:
        """Calculate the Fermi energy of the electron system.

        Returns:
            float: The Fermi energy.
        """
        self.electron_density = self.n_electron / self.lattice.volume["primitive"]
        self.fermi_energy = 0.5 * (3 * np.pi ** 2 * self.electron_density) ** (2/3)
