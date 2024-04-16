import numpy as np
from dataclasses import dataclass

from elphem.const.unit import Energy
from elphem.lattice.empty import EmptyLattice

@dataclass
class EinsteinModel:
    lattice: EmptyLattice
    einstein_temperature: float
    mass: float

    def __post_init__(self):
        if self.einstein_temperature < 0.0:
            raise ValueError("Einstein temperature must be not-negative.")
        if self.mass <= 0.0:
            raise ValueError("Mass must be positive.")
    
    def eigenenergy(self, q: np.ndarray) -> np.ndarray:
        """
        Get phonon eigenenergies.
        
        Arg
            q: A numpy array representing q-vector in the reciprocal space.
            
        Return
            A numpy array representing eigenenergies.
        """
        return np.full(q[..., 0].shape, self.einstein_temperature * Energy.KELVIN["->"])
        
    def grid(self, n_q: np.ndarray) -> np.ndarray:
        """
        Get q-grid.
        
        Arg
            n_q: A numpy array representing the dense of q-vector in the reciprocal space.
        
        Return
            A numpy array (meshgrid) representing q-grid.
        """
        basis = self.lattice.basis["reciprocal"]
        
        grid = np.meshgrid(*[np.linspace(-0.5, 0.5, i) for i in n_q])
        grid = np.array(grid)
        
        x = np.empty(grid[0].shape + (3,))
        for i in range(3):
            x[..., i] = grid[i]

        return x @ basis
    
    def get_dispersion(self, q_names: list[np.ndarray], n_split) -> tuple:
        """
        Get phonon dispersions.
        
        Args
            q_via: Numpy arrays representing special points in the first Brillouin zone.
            n_via: Number of points between special points. The default value is 20.
        """
        x, q, x_special = self.lattice.reciprocal_cell.path(q_names, n_split)
        omega = self.eigenenergy(q)
        
        return x, omega, x_special