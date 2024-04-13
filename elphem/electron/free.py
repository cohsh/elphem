import numpy as np
from dataclasses import dataclass

from elphem.lattice.empty import EmptyLattice

@dataclass
class FreeElectron:
    lattice: EmptyLattice
    electron_per_cell: int
    
    def __post_init__(self):
        if not isinstance(self.lattice, EmptyLattice):
            raise TypeError("The type of first variable must be EmptyLattice.")
        if self.electron_per_cell <= 0:
            raise ValueError("Second variable (number of electrons per unit cell) should be a positive value.")
        
        self.electron_density = self.electron_per_cell / self.lattice.volume["primitive"]

    def fermi_energy(self) -> float:
        """
        Get Fermi energy.
        
        Return
            Fermi energy
        """
        return 0.5 * (3 * np.pi ** 2 * self.electron_density) ** (2/3)

    def eigenenergy(self, k: np.ndarray) -> np.ndarray:
        """
        Get electron eigenenergies.
        
        Args
            k: A numpy array representing vectors in reciprocal space.
        
        Return
            Electron eigenenergies
        """
        return 0.5 * np.linalg.norm(k, axis=k.ndim-1) ** 2 - self.fermi_energy()
    
    def grid(self, n_g: np.ndarray, n_k: np.ndarray) -> tuple:
        """
        Get (G, k)-grid.
        
        Args
            n_g: A numpy array representing the dense of G-grid in reciprocal space.
            n_k: A numpy array representing the dense of k-grid in reciprocal space.
            
        Return
            A tuple containing:
                G-meshgrid
                k-meshgrid
        """
        basis = self.lattice.basis["reciprocal"]
        
        # Combining the grid generation for G and k
        grid_points = [np.arange(-i, i) for i in n_g] + [np.linspace(-0.5, 0.5, i) for i in n_k]
        grid = np.meshgrid(*grid_points)
        
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
        if not hasattr(self.lattice, 'reciprocal_cell') or not callable(self.lattice.reciprocal_cell.path):
            raise AttributeError("The lattice object must have a 'reciprocal_cell' attribute with a 'path' method.")
        if not hasattr(self.lattice, 'grid') or not callable(self.lattice.grid):
            raise AttributeError("The lattice object must have a 'grid' method.")

        x, k, special_x = self.lattice.reciprocal_cell.path(n_via, *k_via)
        g = self.lattice.grid(n_g).reshape(-1, 3)
        eig = np.array([self.eigenenergy(k + gi) for gi in g])
        
        return x, eig, special_x
    
    def get_reciprocal_vector(self, n_band) -> np.ndarray:
        basis = self.lattice.basis["reciprocal"]

        n_1d = np.arange(0, n_band)
        n_3d = np.array(np.meshgrid(n_1d, n_1d, n_1d)).T

        g_norm = np.linalg.norm(n_3d @ basis, axis=-1).round(decimals=5).reshape(-1)
        g_norm_unique = np.unique(g_norm)

        print(g_norm.shape)

        g_dict = {}

        for g_ref in g_norm_unique:
            degeneracy = 0
            for g in g_norm:
                if g == g_ref:
                    degeneracy += 1
            g_dict[g] = degeneracy