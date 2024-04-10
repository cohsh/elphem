import numpy as np
from dataclasses import dataclass

from elphem.elph.self_energy import SelfEnergy

@dataclass
class Spectrum:
    self_energy: SelfEnergy

    def calculate(self, n_g: np.ndarray, n_k: np.ndarray,
                    n_g_inter: np.ndarray, n_q: np.ndarray) -> np.ndarray:
        """
        Calculate 2nd-order Fan self-energies.
        
        Args
            temperature: A temperature in Kelvin.
            g: A numpy array (meshgrid-type) representing G-vector
            k: A numpy array (meshgrid-type) representing k-vector
            n_g: A numpy array representing the dense of intermediate G-vectors
            n_q: A numpy array representing the dense of intermediate q-vectors
            eta: A value of the convergence factor. The default value is 0.01 Hartree.
            
        Return
            A numpy array representing Fan self-energy.
        """
        
        fan_term = self.self_energy.calculate_fan_term(n_g, n_k, n_g_inter, n_q)
        z = self.self_energy.calculate_qp_strength(n_g, n_k, n_g_inter, n_q)
        
        print(fan_term.shape)
        print(z.shape)
        
        # return value.reshape(k[..., 0].shape)