import numpy as np
from dataclasses import dataclass

from elphem.elph.self_energy import SelfEnergy
from elphem.elph.distribution import safe_divide

@dataclass
class Spectrum:
    """Class to calculate the spectral function for electronic states using self-energy components.

    Attributes:
        self_energy (SelfEnergy): An instance of SelfEnergy used for the spectral function calculations.
    """
    
    self_energy: SelfEnergy

    def calculate_with_grid(self, n_k: np.ndarray, n_q: np.ndarray, n_omega: int) -> np.ndarray:
        """
        Calculate the spectral function over a grid of k-points and a range of energies.

        Args:
            n_k (np.ndarray): A numpy array specifying the density of k-grid points in each direction.
            n_q (np.ndarray): A numpy array specifying the density of q-grid points in each direction.
            n_omega (int): Number of points in the energy range for the spectral calculation.

        Returns:
            np.ndarray: The calculated spectral function array over the specified grid and energy range.
        """
        
        g_grid, k_grid = self.self_energy.electron.grid(n_k)
        
        shape_mesh = g_grid[..., 0].shape
        
        g = g_grid.reshape(-1, 3)
        k = k_grid.reshape(-1, 3)

        epsilon = self.self_energy.electron.eigenenergy(k_grid)
        fan_term = np.array([self.self_energy.calculate_fan_term(g_i, k_i, n_q) for g_i, k_i in zip(g, k)]).reshape(shape_mesh)
        qp_strength = np.array([self.self_energy.calculate_qp_strength(g_i, k_i, n_q) for g_i, k_i in zip(g, k)]).reshape(shape_mesh)

        coeff = - qp_strength / np.pi
        numerator = qp_strength * fan_term.imag
        
        omegas = np.linspace(0.0, 10.0, n_omega)

        spectrum = np.zeros((np.prod(n_k), n_omega))
                
        count = 0
        for omega in omegas:
            denominator = (
                (omega - epsilon - fan_term.real) ** 2
                + (qp_strength * fan_term.imag) ** 2
                )
            fraction = safe_divide(coeff * numerator, denominator)
            spectrum[..., count] = np.nansum(fraction, axis=0)
            
            count += 1
        
        return spectrum
    
    def calculate_with_path(self, k_names: list[str], n_split: int,
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
        
        g = self.self_energy.electron.g
        
        x, k, special_x = self.self_energy.lattice.reciprocal_cell.path(k_names, n_split)
        epsilon = np.array([self.self_energy.electron.eigenenergy(k + g_i) for g_i in g])

        shape_return = epsilon.shape

        fan_term = np.zeros(shape_return, dtype='complex128')
        qp_strength = np.zeros(shape_return)

        for i in range(self.self_energy.electron.n_band):
            fan_term[i] = np.array([self.self_energy.calculate_fan_term(g[i], k_i, n_q) for k_i in k])
            qp_strength[i] = np.array([self.self_energy.calculate_qp_strength(g[i], k_i, n_q) for k_i in k])

        coeff = - qp_strength / np.pi
        numerator = qp_strength * fan_term.imag

        omegas = np.linspace(range_omega[0], range_omega[1], n_omega)
        spectrum = np.zeros(fan_term[0].shape + omegas.shape)
                
        count = 0
        for omega in omegas:
            denominator = (omega - epsilon - fan_term.real) ** 2 + (qp_strength * fan_term.imag) ** 2
            fraction = safe_divide(coeff * numerator, denominator)

            spectrum[..., count] = np.nansum(fraction, axis=0)
            
            count += 1
        
        return x, omegas, spectrum, special_x