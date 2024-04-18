import numpy as np
from dataclasses import dataclass
from elphem.lattice import *

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
    
    def __post_init__(self):
        """Validate and initialize the FreeElectron model."""
        if not isinstance(self.lattice, Lattice):
            raise TypeError("The type of first variable must be EmptyLattice.")
        if self.n_electron <= 0:
            raise ValueError("Second variable (number of electrons per unit cell) should be a positive value.")
        
        self.set_fermi_energy()
        self.reciprocal_vectors = self.lattice.get_reciprocal_vectors(self.n_band)

    def get_eigenenergy(self, k: np.ndarray) -> np.ndarray:
        """Calculate the electron eigenenergies at wave vector k.

        Args:
            k (np.ndarray): A numpy array representing vectors in reciprocal space.

        Returns:
            np.ndarray: The electron eigenenergies at each wave vector.
        """

        eigenenergy = 0.5 * np.linalg.norm(k, axis=-1) ** 2 - self.fermi_energy

        return eigenenergy

    def get_gk_grid(self, n_k: np.ndarray) -> tuple:
        """Generate a (G, k)-grid for electron states calculation.

        Args:
            n_k (np.ndarray): A numpy array specifying the density of k-grid points in each direction of reciprocal space.

        Returns:
            tuple: A tuple containing G-meshgrid and k-meshgrid for electron state calculations.
        """
        if np.array(n_k).shape != (3,):
            raise ValueError("Shape of n_k should be (3,).")
        
        k_grid = self.lattice.get_grid(*n_k)

        k_return = np.tile(k_grid.mesh, (self.n_band, 1, 1))
        g_return = np.repeat(self.reciprocal_vectors[:, np.newaxis, :], len(k_grid.mesh), axis=1)

        return g_return, k_return

    def get_band_structure(self, k_names: list[np.ndarray], n_split: int) -> tuple:
        """Calculate the electronic band structures along the specified path in reciprocal space.

        Args:
            k_names (list[np.ndarray]): A list of special points names defining the path.
            n_split (int): Number of points between each special point to compute the band structure.

        Returns:
            tuple: A tuple containing x-coordinates for plotting, eigenenergy values, and x-coordinates of special points.
        """
        x, k, special_x = self.lattice.reciprocal_cell.get_path(k_names, n_split)
        
        eigenenergy = np.array([self.get_eigenenergy(k + g_i) for g_i in self.reciprocal_vectors])
        
        return x, eigenenergy, special_x

    def set_fermi_energy(self) -> None:
        """Calculate the Fermi energy of the electron system.

        Returns:
            float: The Fermi energy.
        """
        electron_density = self.n_electron / self.lattice.volume["primitive"]
        
        self.fermi_energy = 0.5 * (3 * np.pi ** 2 * electron_density) ** (2/3)
