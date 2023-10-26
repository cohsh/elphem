import numpy as np
from dataclasses import dataclass

from elphem.lattice.empty import EmptyLattice

@dataclass
class FreeElectron:
    lattice: EmptyLattice
    electron_per_cell: int
    
    def __post_init__(self):
        self.electron_density = self.electron_per_cell / self.lattice.volume["primitive"]

    def fermi_energy(self) -> float:
        fermi_energy = 0.5 * (3 * np.pi ** 2 * self.electron_density) ** (2/3)
        return fermi_energy

    def eigenenergy(self, k: np.ndarray) -> np.ndarray:
        eigenenergy = 0.5 * np.linalg.norm(k, axis=k.ndim-1) ** 2 - self.fermi_energy()
        return eigenenergy

    @staticmethod    
    def velocity(k: np.ndarray) -> np.ndarray:
        velocity = k
        return velocity
    
    def grid(self, n_g: np.ndarray, n_k: np.ndarray) -> np.ndarray:
        basis = self.lattice.basis["reciprocal"]
        
        grid = np.meshgrid(*[np.arange(-i, i) for i in n_g], *[np.linspace(-0.5, 0.5, i) for i in n_k])
        grid = np.array(grid)
        
        grid_set = []
        j = 0
        
        for i in range(2):
            x = grid[j:j+3]
            y = np.empty(x[0].shape + (3,))
            for k in range(3):
                y[..., k] = x[k]

            grid_set.append(y @ basis)
            j += 3

        return tuple(grid_set)
    
    def get_band_structure(self, n_g: np.ndarray, *k_via: list[np.ndarray], n_via=20) -> tuple:
        """
        Calculate the electronic band structures.

        Args
            n_g: A numpy array representing the grid in reciprocal space.
            k_via: A list of numpy arrays representing the special points in reciprocal space.
            n_via: Number of points between the special points. Default is 20.
        
        Return
            A tuple containing:
                x: x-coordinates for plotting
                eig: Eigenenergy values
                special_x: x-coordinates of special points
        """
        x, k, special_x = self.lattice.reciprocal_cell.path(n_via, *k_via)
        g = self.lattice.grid(n_g).reshape(-1, 3)
        eig = np.array([self.eigenenergy(k + gi) for gi in g])
        
        return x, eig, special_x