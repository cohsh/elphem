import numpy as np
from dataclasses import dataclass

@dataclass
class Grid:
    basis: np.ndarray
    n_x: int
    n_y: int
    n_z: int
    
    def __post_init__(self):
        self.n_mesh = self.n_x * self.n_y * self.n_z
        self.generate()
    
    def generate(self) -> None:
        """Generate empty k-grid"""
        self.mesh = np.empty((self.n_mesh, 3))

    def align(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        aligned_k = np.array(np.meshgrid(x, y, z, indexing='ij')).T.reshape(-1, 3) @ self.basis
        
        return aligned_k

class SimpleGrid(Grid):
    def __init__(self, basis, n_x, n_y, n_z):
        super().__init__(basis, n_x, n_y, n_z)

    def generate(self) -> None:
        """Generate simple k-grid"""
        
        x = np.linspace(-0.5, 0.5, self.n_x)
        y = np.linspace(-0.5, 0.5, self.n_y)
        z = np.linspace(-0.5, 0.5, self.n_z)

        self.mesh = self.align(x, y, z)


class MonkhorstPackGrid(Grid):
    def __init__(self, basis, n_x, n_y, n_z):
        super().__init__(basis, n_x, n_y, n_z)

    def generate(self) -> None:
        """Generate Monkhorst and Pack grid"""

        r_x = np.arange(1, self.n_x + 1)
        r_y = np.arange(1, self.n_y + 1)
        r_z = np.arange(1, self.n_z + 1)

        x = (2 * r_x - self.n_x - 1) / ( 2 * self.n_x )
        y = (2 * r_y - self.n_y - 1) / ( 2 * self.n_y )
        z = (2 * r_z - self.n_z - 1) / ( 2 * self.n_z )

        self.mesh = self.align(x, y, z)