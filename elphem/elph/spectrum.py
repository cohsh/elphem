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

    def get_with_grid(self, n_k: np.ndarray, n_q: np.ndarray, n_omega: int) -> np.ndarray:
        """
        Calculate the spectral function over a grid of k-points and a range of energies.

        Args:
            n_k (np.ndarray): A numpy array specifying the density of k-grid points in each direction.
            n_q (np.ndarray): A numpy array specifying the density of q-grid points in each direction.
            n_omega (int): Number of points in the energy range for the spectral calculation.

        Returns:
            np.ndarray: The calculated spectral function array over the specified grid and energy range.
        """
        
        g_grid, k_grid = self.electron_phonon.electron.get_gk_grid(n_k)
        
        shape_mesh = g_grid[..., 0].shape
        
        g = g_grid.reshape(-1, 3)
        k = k_grid.reshape(-1, 3)

        electron_eigenenergy = self.electron_phonon.electron.get_eigenenergy(k_grid)
        
        results = [self.electron_phonon.get_self_energy_and_coupling_strength(g_i, k_i, n_q) for g_i, k_i in zip(g, k)]

        self_energy = np.array([result[0] for result in results]).reshape(shape_mesh)
        qp_strength = self.electron_phonon.get_qp_strength(np.array([result[1] for result in results])).reshape(shape_mesh)

        coefficient = - qp_strength / np.pi
        numerator = qp_strength * self_energy.imag
        
        omegas = np.linspace(0.0, 10.0, n_omega)

        spectrum = np.zeros((np.prod(n_k), n_omega))
                
        count = 0
        for omega in omegas:
            denominator = (
                (omega - electron_eigenenergy - self_energy.real) ** 2
                + (qp_strength * self_energy.imag) ** 2
                )
            fraction = safe_divide(coefficient * numerator, denominator)
            spectrum[..., count] = np.nansum(fraction, axis=0)
            
            count += 1
        
        return spectrum
    
    def get_with_path(self, k_names: list[str], n_split: int,
                    n_q: np.ndarray, n_omega: int, range_omega: list[float]) -> tuple:
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
        electron_eigenenergy = np.array([self.electron_phonon.electron.get_eigenenergy(k + g_i) for g_i in g])

        shape_return = electron_eigenenergy.shape

        self_energy = np.zeros(shape_return, dtype='complex128')
        qp_strength = np.zeros(shape_return)

        for i in range(self.electron_phonon.electron.n_band):
            results = [self.electron_phonon.get_self_energy_and_coupling_strength(g[i], k_i, n_q) for k_i in k]
            self_energy[i] = np.array([result[0] for result in results])
            qp_strength[i] = self.electron_phonon.get_qp_strength(np.array([result[1] for result in results]))

        coefficient = - qp_strength / np.pi
        numerator = qp_strength * self_energy.imag

        omegas = np.linspace(range_omega[0], range_omega[1], n_omega)
        spectrum = np.zeros(self_energy[0].shape + omegas.shape)
                
        count = 0
        for omega in omegas:
            denominator = (omega - electron_eigenenergy - self_energy.real) ** 2 + (qp_strength * self_energy.imag) ** 2
            fraction = safe_divide(coefficient * numerator, denominator)

            spectrum[..., count] = np.nansum(fraction, axis=0)
            
            count += 1
        
        return x, omegas, spectrum, special_x