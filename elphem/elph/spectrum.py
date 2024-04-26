import numpy as np
from dataclasses import dataclass

from elphem.elph.electron_phonon import ElectronPhonon
from elphem.elph.distribution import safe_divide

@dataclass
class Spectrum:
    """Class to calculate the spectral function for electronic states using self-energy components.

    Attributes:
        self_energy (SelfEnergy): An instance of SelfEnergy used for the spectral function calculations.
    """

    electron_phonon: ElectronPhonon

    def get_with_path(self, k_names: list[str], n_split: int, n_omega: int, range_omega: list[float]) -> tuple:
        """
        Calculate the spectral function along a specified path in the Brillouin zone.

        Args:
            k_names (list[str]): List of special points defining the path through the Brillouin zone.
            n_split (int): Number of points between each special point.
            n_q (np.ndarray): A numpy array specifying the density of q-grid points in each direction.
            n_omega (int): Number of points in the energy range.
            range_omega (list[float]): The range of energy values over which to calculate the spectrum.

        Returns:
            tuple: A tuple containing the path x-coordinates, energy values, the calculated spectrum, and x-coordinates of special points.
        """
        
        g = self.electron_phonon.electron.reciprocal_vectors
        
        x, k, special_x = self.electron_phonon.electron.lattice.reciprocal_cell.get_path(k_names, n_split)
        eig = np.array([self.electron_phonon.electron.get_eigenenergy(k + g_i) for g_i in g])

        omega_array = np.linspace(range_omega[0], range_omega[1], n_omega)
        spectrum = np.zeros(eig[0].shape + omega_array.shape)
                
        self_energy = self.electron_phonon.get_self_energy(omega_array, k)
        
        numerator = - self_energy.imag / np.pi
        denominator = (omega_array - eig - self_energy.real) ** 2 + self_energy.imag ** 2

        fraction = safe_divide(numerator, denominator)

        spectrum[..., count] = np.nansum(fraction, axis=0)

        return x, omegas, spectrum, special_x