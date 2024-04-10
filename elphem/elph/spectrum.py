import numpy as np
from dataclasses import dataclass

from elphem.elph.self_energy import SelfEnergy

@dataclass
class Spectrum:
    self_energy: SelfEnergy

    def calculate(self, n_g: np.ndarray, n_k: np.ndarray,
                    n_g_inter: np.ndarray, n_q: np.ndarray, n_omega: int) -> np.ndarray:
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
        
        fan_term = self.self_energy.calculate_fan_term(n_g, n_k, n_g_inter, n_q)
        qp_strength = self.self_energy.calculate_qp_strength(n_g, n_k, n_g_inter, n_q)
        
        g, k = self.self_energy.electron.grid(n_g, n_k)
        epsilon = self.self_energy.electron.eigenenergy(k + g)

        coeff = - qp_strength / np.pi
        numerator = qp_strength * fan_term.imag
        
        omegas = np.linspace(0.0, 10.0, n_omega)

        spectrum = np.zeros(fan_term.shape[0:3] + omegas.shape)
        
        count = 0
        for omega in omegas:
            denominator = (
                (omega - epsilon - fan_term.real) ** 2
                + (qp_strength * fan_term.imag) ** 2
                )
            spectrum[..., count] = np.nansum(coeff * numerator / denominator, axis=(3,4,5))
            
            count += 1
        
        return spectrum