import numpy as np
from dataclasses import dataclass
from elphem.lattice.empty import EmptyLattice

@dataclass
class FreeElectron:
    lattice: EmptyLattice
    n_band: int
    electron_per_cell: int
    
    def __post_init__(self):
        if not isinstance(self.lattice, EmptyLattice):
            raise TypeError("The type of first variable must be EmptyLattice.")
        if self.electron_per_cell <= 0:
            raise ValueError("Second variable (number of electrons per unit cell) should be a positive value.")
        
        self.electron_density = self.electron_per_cell / self.lattice.volume["primitive"]
        self.g = self.get_reciprocal_vector()

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

        return 0.5 * np.linalg.norm(k, axis=-1) ** 2 - self.fermi_energy()
    
    def grid(self, n_k: np.ndarray) -> tuple:
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
        
        k_x = np.linspace(-0.5, 0.5, n_k[0])
        k_y = np.linspace(-0.5, 0.5, n_k[1])
        k_z = np.linspace(-0.5, 0.5, n_k[2])
        k = np.array(np.meshgrid(k_x, k_y, k_z)).T.reshape(-1, 3) @ basis
        
        shape_return = (self.n_band, len(k), 3)
        
        g_grid = np.zeros(shape_return)
        k_grid = np.zeros(shape_return)
        
        for n in range(self.n_band):
            for i in range(len(k)):
                g_grid[n,i] = self.g[n]
                k_grid[n,i] = k[i]

        return g_grid, k_grid
    
    def get_band_structure(self, *k_via: list[np.ndarray], n_via=20) -> tuple:
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
        
        eigenenergy = np.array([self.eigenenergy(k + g_i) for g_i in self.g])
        
        return x, eigenenergy, special_x
    
    def get_reciprocal_vector(self) -> np.ndarray:
        basis = self.lattice.basis["reciprocal"]

        n_1d = np.arange(0, np.cbrt(self.n_band))
        n_3d = np.array(np.meshgrid(n_1d, n_1d, n_1d)).T.reshape(-1, 3)

        g = n_3d @ basis
        g_norm = np.linalg.norm(g, axis=-1).round(decimals=5)
        g_norm_unique = np.unique(g_norm)

        g_list = []

        for g_ref in g_norm_unique:
            count = 0
            for g_compare in g_norm:
                if g_compare == g_ref:
                    g_list.append(g[count])
                count += 1

        return np.array(g_list[0:self.n_band])