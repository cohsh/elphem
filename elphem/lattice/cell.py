import numpy as np

class Cell3D:
    """Class for 3D cell
    Attributes:
        n_dim (int): Number of dimension
        basis (np.ndarray): A numpy array of basis
    """
    def __init__(self):
        self.n_dim = 3
        self.basis = None
    
    def calculate_volume(self) -> float:
        """Calculates the volume of the cell.

        Returns:
            float: The volume.
        """
        return np.dot(self.basis[0], np.cross(self.basis[1], self.basis[2]))

class Cell2D:
    """Class for 2D cell
    Attributes:
        n_dim (int): Number of dimension
        basis (np.ndarray): A numpy array of basis
    """
    def __init__(self):
        self.n_dim = 2
        self.basis = None
    
    def calculate_volume(self) -> float:
        """Calculates the volume of the cell.

        Returns:
            float: The volume.
        """
        return np.linalg.norm(np.cross(self.basis[0], self.basis[1]))

class Cell1D:
    """Class for 1D cell
    Attributes:
        n_dim (int): Number of dimension
        basis (np.ndarray): A numpy array of basis
    """

    def __init__(self):
        self.n_dim = 1
        self.basis = None
    
    def calculate_volume(self) -> float:
        """Calculates the volume of the cell.

        Returns:
            float: The volume.
        """
        return np.linalg.norm(self.basis)