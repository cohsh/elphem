import numpy as np
from dataclasses import dataclass

from elphem.elph.self_energy import SelfEnergy

@dataclass
class EPR:
    """A class to calculate the 2nd-order Fan self-energies using the self-energy module.

    Attributes:
        self_energy (SelfEnergy): An instance of SelfEnergy to use for calculations.
    """
    self_energy: SelfEnergy

    def calculate_with_grid(self, n_k: np.ndarray, n_q: np.ndarray) -> np.ndarray:
        """
        Calculate 2nd-order Fan self-energies over a grid of k-points and q-points.

        Args:
            n_k (np.ndarray): A numpy array specifying the density of k-grid points in each direction.
            n_q (np.ndarray): A numpy array specifying the density of q-grid points in each direction.

        Returns:
            tuple: A tuple containing the eigenenergies array and the Fan self-energy array calculated over the grid.
        """
        
        g_grid, k_grid = self.self_energy.electron.grid(n_k)
        
        shape_mesh = g_grid[..., 0].shape
        
        g = g_grid.reshape(-1, 3)
        k = k_grid.reshape(-1, 3)

        eig = self.self_energy.electron.eigenenergy(k_grid)
        fan_term = np.array([self.self_energy.calculate_fan_term(g_i, k_i, n_q) for g_i, k_i in zip(g, k)]).reshape(shape_mesh)
        
        epr = fan_term.real
        
        return eig, epr
    
    def calculate_with_path(self, k_names: list[str], n_split: int, n_q: np.ndarray) -> tuple:
        """
        Calculate 2nd-order Fan self-energies along a specified path in the Brillouin zone.

        Args:
            k_names (list[str]): A list of special points names defining the path through the Brillouin zone.
            n_split (int): Number of points between each special point to compute the dispersion.
            n_q (np.ndarray): A numpy array specifying the density of q-grid points in each direction.

        Returns:
            tuple: A tuple containing x-coordinates for plotting, eigenenergies, Fan self-energies, and x-coordinates of special points.
        """
        
        g = self.self_energy.electron.g
        
        x, k, special_x = self.self_energy.lattice.reciprocal_cell.path(k_names, n_split)
        eig = np.array([self.self_energy.electron.eigenenergy(k + g_i) for g_i in g])

        shape_return = eig.shape

        fan_term = np.zeros(shape_return, dtype='complex128')

        for i in range(self.self_energy.electron.n_band):
            fan_term[i] = np.array([self.self_energy.calculate_fan_term(g[i], k_i, n_q) for k_i in k])

        epr = fan_term.real
        
        return x, eig, epr, special_x