import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

from elphem.const.unit import Energy
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
    
    def save_band(self, file_name, n_g: np.ndarray, k_names: list, *k_via: list[np.ndarray], 
                    ylim=[], n_via=20) -> None:
        x, k, special_x = self.lattice.reciprocal_cell.path(n_via, *k_via)
    
        g = self.lattice.grid(n_g)
        g = g.reshape(int(g.size/3) ,3)

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        for i in range(len(g)):
            eig = self.eigenenergy(k + g[i])
            ax.plot(x, eig * Energy.eV["<-"], color="tab:blue")
        
        for x0 in special_x:
            ax.axvline(x=x0, color="black", linewidth=0.3)
        
        ax.set_xticks(special_x)
        ax.set_xticklabels(k_names)
        ax.set_ylabel("Energy ($\mathrm{eV}$)")
        if ylim != []:
            ax.set_ylim(ylim)

        fig.savefig(file_name)