import numpy as np

class Cell3D:
    def __init__(self):
        self.n_dim = 3
        self.basis = None
    
    def calculate_volume(self) -> float:
        """Calculates the volume of the cell.

        Returns:
            float: The volume calculated using the determinant of the basis vectors.
        """
        return np.dot(self.basis[0], np.cross(self.basis[1], self.basis[2]))

class Cell2D:
    def __init__(self):
        self.n_dim = 2
        self.basis = None
    
    def calculate_volume(self) -> float:
        return np.linalg.norm(np.cross(self.basis[0], self.basis[1]))

class Cell1D:
    def __init__(self):
        self.n_dim = 1
        self.basis = None
    
    def calculate_volume(self) -> float:
        return np.linalg.norm(self.basis)