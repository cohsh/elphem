import numpy as np
from dataclasses import dataclass

from elphem.const.unit import Energy
from elphem.lattice.empty import EmptyLattice

@dataclass
class DebyeModel:
    """Models the phononic properties of a lattice using the Debye model.

    Attributes:
        lattice (EmptyLattice): The crystal lattice on which the model is applied.
        debye_temperature (float): The Debye temperature of the lattice material.
        number_of_atom (float): The number of atoms per primitive cell.
        mass (float): The mass of the crystal's atoms.
    """
    lattice: EmptyLattice
    debye_temperature: float
    number_of_atom: float
    mass: float

    def __post_init__(self):
        """Validate initial model parameters."""
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
        """Calculate the speed of sound in the lattice based on Debye model.

        Returns:
            float: The speed of sound in Hartree atomic units.
        """
        debye_frequency = self.debye_temperature * Energy.KELVIN["->"]

        return debye_frequency * (6.0 * np.pi ** 2 * self.number_density) ** (-1.0/3.0)
    
    def eigenenergy(self, q: np.ndarray) -> np.ndarray:
        """Calculate phonon eigenenergies at wave vector q.

        Args:
            q (np.ndarray): A numpy array representing vectors in reciprocal space.

        Returns:
            np.ndarray: The phonon eigenenergies at each wave vector.
        """
        return self.speed_of_sound() * np.linalg.norm(q, axis=q.ndim-1)
    
    def eigenvector(self, q: np.ndarray) -> np.ndarray:
        """Calculate phonon eigenvectors at wave vector q.

        Args:
            q (np.ndarray): A numpy array representing vectors in reciprocal space.

        Returns:
            np.ndarray: The phonon eigenvectors at each wave vector, represented as complex numbers.
        """
        q_norm = np.linalg.norm(q, axis=q.ndim-1)

        q_normalized = np.divide(q, q_norm[:, np.newaxis], out=np.zeros_like(q), where=q_norm[:, np.newaxis] != 0)
        return 1.0j * q_normalized

    
    def grid(self, n_q: np.ndarray) -> np.ndarray:
        """Generate a q-grid for phonon calculations.

        Args:
            n_q (np.ndarray): A numpy array specifying the density of q-grid points in each direction of reciprocal space.

        Returns:
            q (np.ndarray): A meshgrid representing the q-grid in reciprocal space.
        """

        q = self.lattice.reciprocal_cell.monkhorst_pack_grid(*n_q)

        return q
    
    def get_dispersion(self, q_names: list[np.ndarray], n_split) -> tuple:
        """Calculate the phonon dispersion curves along specified paths in reciprocal space.

        Args:
            q_names (list[np.ndarray]): List of special points defining the path through the Brillouin zone.
            n_split (int): Number of points between each special point to compute the dispersion curve.

        Returns:
            tuple: A tuple containing the x-coordinates for plotting, omega (eigenenergy values), and x-coordinates of special points.
        """
        x, q, x_special = self.lattice.reciprocal_cell.path(q_names, n_split)
        omega = self.eigenenergy(q)
        
        return x, omega, x_special