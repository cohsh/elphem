from dataclasses import dataclass

from elphem.elph.self_energy import SelfEnergy

@dataclass
class Spectrum:
    self_energy: SelfEnergy

    def calculate(self, g: np.ndarray, k: np.ndarray,
                    n_g: np.ndarray, n_q: np.ndarray) -> np.ndarray:
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
        
        g_reshaped = g.reshape(-1, 3)
        k_reshaped = k.reshape(-1, 3)
        
        fan_term = np.array([self.self_energy.calculate_fan_term(g_i, k_i, n_g, n_q) for g_i, k_i in zip(g_reshaped, k_reshaped)])
        z = np.array([self.self_energy.calculate_qp_strength(g_i, k_i, n_g, n_q) for g_i, k_i in zip(g_reshaped, k_reshaped)])
        
        # return value.reshape(k[..., 0].shape)
