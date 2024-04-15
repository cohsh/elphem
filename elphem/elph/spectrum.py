import numpy as np
from dataclasses import dataclass

from elphem.elph.self_energy import SelfEnergy
from elphem.elph.distribution import safe_divide

@dataclass
class Spectrum:
    self_energy: SelfEnergy

    def calculate_with_grid(self, n_k: np.ndarray, n_q: np.ndarray, n_omega: int) -> np.ndarray:
        """
        Calculate 2nd-order Fan self-energies.
        
        Args
            temperature: A temperature in Kelvin.
            n_g: A numpy array (meshgrid-type) representing G-vector
            n_k: A numpy array (meshgrid-type) representing k-vector
            n_g: A numpy array representing the dense of intermediate G-vectors
            n_q: A numpy array representing the dense of intermediate q-vectors
            
        Return
            A numpy array representing Fan self-energy.
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
    
    def calculate_with_path(self, k_names: list[np.ndarray], n_split: int,
                    n_q: np.ndarray, n_omega: int, range_omega: list) -> tuple:
        """
        Calculate 2nd-order Fan self-energies.
        
        Args
            temperature: A temperature in Kelvin.
            n_g: A numpy array (meshgrid-type) representing G-vector
            n_k: A numpy array (meshgrid-type) representing k-vector
            n_g: A numpy array representing the dense of intermediate G-vectors
            n_q: A numpy array representing the dense of intermediate q-vectors
            
        Return
            A numpy array representing Fan self-energy.
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