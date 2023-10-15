import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

from elphem.const.unit import Energy, Prefix
from elphem.lattice.empty import EmptyLattice

@dataclass
class DebyeModel:
    lattice: EmptyLattice
    debye_temperature: float
    number_of_atom: float
    mass: float

    def __post_init__(self):
        self.number_density = self.number_of_atom / self.lattice.volume["primitive"]
        self.speed = self.speed_of_sound()

    def speed_of_sound(self) -> float:
        debye_frequency = self.debye_temperature * Energy.kelvin["->"]
        speed_of_sound = debye_frequency * (6.0 * np.pi ** 2 * self.number_density) ** (-1.0/3.0)
        return speed_of_sound
    
    def eigenenergy(self, q: np.ndarray) -> np.ndarray:
        eigenenergy = self.speed_of_sound() * np.linalg.norm(q, axis=q.ndim-1)
        return eigenenergy
    
    def eigenvector(self, q: np.ndarray) -> np.ndarray:
        q_norm = np.linalg.norm(q, axis=q.ndim-1)
        
        q_normalized = np.empty(q.shape)
        for i in range(3):
            q_normalized[..., i] = 1.0j * q[..., i] / q_norm

        return q_normalized
    
    def grid(self, n_q: np.ndarray) -> np.ndarray:
        basis = self.lattice.basis["reciprocal"]
        
        grid = np.meshgrid(*[np.linspace(-0.5, 0.5, i) for i in n_q])
        grid = np.array(grid)
        
        x = np.empty(grid[0].shape + (3,))
        for i in range(3):
            x[..., i] = grid[i]

        return x @ basis
    
    def save_dispersion(self, file_name: str, q_names: list, *q_via: list[np.ndarray], n_via=20) -> None:
        x, q, special_x = self.lattice.reciprocal_cell.path(n_via, *q_via)

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        omega = self.eigenenergy(q)
        ax.plot(x, omega * Energy.eV["<-"] / Prefix.milli, color="tab:blue")
        
        for x0 in special_x:
            ax.axvline(x=x0, color="black", linewidth=0.3)
        
        ax.set_xticks(special_x)
        ax.set_xticklabels(q_names)
        ax.set_ylabel("Energy ($\mathrm{meV}$)")

        fig.savefig(file_name)