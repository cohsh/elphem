import numpy as np
from dataclasses import dataclass

from elphem.elph.self_energy import SelfEnergy
from elphem.elph.distribution import safe_divide

@dataclass
class EPR:
    self_energy: SelfEnergy

    def calculate_with_grid(self, n_k: np.ndarray, n_q: np.ndarray) -> np.ndarray:
        """
        Calculate 2nd-order Fan self-energies.
        
        Args
            temperature: A temperature in Kelvin.
            n_g: A numpy array (meshgrid-type) representing G-vector
            n_k: A numpy array (meshgrid-type) representing k-vector
            n_g: A numpy array representing the dense of intermediate G-vectors
            n_q: A numpy array representing the dense of intermediate q-vectors
            
        Return
            A numpy array representing Fan self-energy.
        """
        
        g_grid, k_grid = self.self_energy.electron.grid(n_k)
        
        shape_mesh = g_grid[..., 0].shape
        
        g = g_grid.reshape(-1, 3)
        k = k_grid.reshape(-1, 3)

        eig = self.self_energy.electron.eigenenergy(k_grid)
        fan_term = np.array([self.self_energy.calculate_fan_term(g_i, k_i, n_q) for g_i, k_i in zip(g, k)]).reshape(shape_mesh)
        
        epr = fan_term.real
        
        return eig, epr
    
    def calculate_with_path(self, k_names: list[np.ndarray], n_split: int, n_q: np.ndarray) -> tuple:
        """
        Calculate 2nd-order Fan self-energies.
        
        Args
            temperature: A temperature in Kelvin.
            n_g: A numpy array (meshgrid-type) representing G-vector
            n_k: A numpy array (meshgrid-type) representing k-vector
            n_g: A numpy array representing the dense of intermediate G-vectors
            n_q: A numpy array representing the dense of intermediate q-vectors
            
        Return
            A numpy array representing Fan self-energy.
        """
        
        g = self.self_energy.electron.g
        
        x, k, special_x = self.self_energy.lattice.reciprocal_cell.path(*k_names, n_split)
        eig = np.array([self.self_energy.electron.eigenenergy(k + g_i) for g_i in g])

        shape_return = eig.shape

        fan_term = np.zeros(shape_return, dtype='complex128')

        for i in range(self.self_energy.electron.n_band):
            fan_term[i] = np.array([self.self_energy.calculate_fan_term(g[i], k_i, n_q) for k_i in k])

        epr = fan_term.real
        
        return x, eig, epr, special_x