import numpy as np
from dataclasses import dataclass

from elphem.const.unit import Energy
from elphem.lattice.lattice import Lattice

@dataclass
class DebyePhonon:
    """Models the phononic properties of a lattice using the Debye model.

    Attributes:
        lattice (EmptyLattice): The crystal lattice on which the model is applied.
        debye_temperature (float): The Debye temperature of the lattice material.
        number_of_atom (float): The number of atoms per primitive cell.
        mass (float): The mass of the crystal's atoms.
    """

    lattice: Lattice
    debye_temperature: float

    def __post_init__(self):
        """Validate initial model parameters."""
        if self.debye_temperature < 0.0:
            raise ValueError("Debye temperature must be not-negative.")

        self.set_speed_of_sound()

    def get_eigenenergy(self, q: np.ndarray) -> np.ndarray:
        """Calculate phonon eigenenergies at wave vector q.

        Args:
            q (np.ndarray): A numpy array representing vectors in reciprocal space.

        Returns:
            np.ndarray: The phonon eigenenergies at each wave vector.
        """

        eigenenergy = self.speed_of_sound * np.linalg.norm(q, axis=-1)
        
        return eigenenergy

    def get_eigenvector(self, q: np.ndarray) -> np.ndarray:
        """Calculate phonon eigenvectors at wave vector q.

        Args:
            q (np.ndarray): A numpy array representing vectors in reciprocal space.

        Returns:
            np.ndarray: The phonon eigenvectors at each wave vector, represented as complex numbers.
        """

        q_norm = np.linalg.norm(q, axis=-1)

        eigenvector = 1.0j * np.divide(q, q_norm[:, np.newaxis], out=np.zeros_like(q), where=q_norm[:, np.newaxis] != 0)

        return eigenvector

    def get_dispersion(self, q_names: list[np.ndarray], n_split) -> tuple:
        """Calculate the phonon dispersion curves along specified paths in reciprocal space.

        Args:
            q_names (list[np.ndarray]): List of special points defining the path through the Brillouin zone.
            n_split (int): Number of points between each special point to compute the dispersion curve.

        Returns:
            tuple: A tuple containing the x-coordinates for plotting, omega (eigenenergy values), and x-coordinates of special points.
        """

        x, q, x_special = self.lattice.reciprocal_cell.get_path(q_names, n_split)
        omega = self.get_eigenenergy(q)
        
        return x, omega, x_special

    def set_speed_of_sound(self) -> None:
        """Calculate the speed of sound in the lattice based on Debye model.

        Returns:
            float: The speed of sound in Hartree atomic units.
        """
        try:
            number_density = self.lattice.n_atoms / self.lattice.volume["primitive"]
        except ZeroDivisionError:
            ValueError("Lattice volume must be positive.")


        debye_frequency = self.debye_temperature * Energy.KELVIN["->"]

        self.speed_of_sound = debye_frequency * (6.0 * np.pi ** 2 * number_density) ** (-1.0/3.0)