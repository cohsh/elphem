import numpy as np
from dataclasses import dataclass

from elphem.const.unit import Energy
from elphem.lattice.empty import EmptyLattice

@dataclass
class DebyeModel:
    lattice: EmptyLattice
    debye_temperature: float
    number_of_atom: float
    mass: float

    def __post_init__(self):
        if self.number_of_atom <= 0:
            raise ValueError("Number of atoms must be positive.")
        if self.debye_temperature < 0.0:
            raise ValueError("Debye temperature must be not-negative.")
        if self.mass <= 0.0:
            raise ValueError("Mass must be positive.")

        try:
            self.number_density = self.number_of_atom / self.lattice.volume["primitive"]
        except ZeroDivisionError:
            ValueError("Lattice volume must be positive.")

        self.speed = self.speed_of_sound()

    def speed_of_sound(self) -> float:
        """
        Get the speed of sound.
        
        Return
            Speed of sound in Hartee atomic unit.
        """
        debye_frequency = self.debye_temperature * Energy.KELVIN["->"]

        return debye_frequency * (6.0 * np.pi ** 2 * self.number_density) ** (-1.0/3.0)
    
    def eigenenergy(self, q: np.ndarray) -> np.ndarray:
        """
        Get phonon eigenenergies.
        
        Arg
            q: A numpy array representing q-vector in the reciprocal space.
            
        Return
            A numpy array representing eigenenergies.
        """
        return self.speed_of_sound() * np.linalg.norm(q, axis=q.ndim-1)
    
    def eigenvector(self, q: np.ndarray) -> np.ndarray:
        """
        Get phonon eigenvector
        
        Arg
            q: A numpy array representing q-vector in the reciprocal space.
            
        Return
            A numpy array representing eigenvectors.
        """
        q_norm = np.linalg.norm(q, axis=q.ndim-1)

        q_normalized = np.divide(q, q_norm[:, np.newaxis], out=np.zeros_like(q), where=q_norm[:, np.newaxis] != 0)
        return 1.0j * q_normalized

    
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