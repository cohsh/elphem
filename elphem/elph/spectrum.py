from dataclasses import dataclass

from elphem.elph.self_energy import SelfEnergy2nd

@dataclass
class Spectrum:
    self_energy: SelfEnergy2nd
    
    def validate_inputs(self, temperature, eta):
        if not isinstance(temperature, (int, float)) or temperature < 0:
            raise ValueError("Temperature must be a not-negative number.")
        if not isinstance(eta, (int, float)):
            raise ValueError("eta must be a number.")

    def calculate(self, temperature: float, g: np.ndarray, k: np.ndarray,
                    n_g: np.ndarray, n_q: np.ndarray, eta=0.01) -> np.ndarray:
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
        self.validate_inputs(temperature, eta)
        g_reshaped = g.reshape(-1, 3)
        k_reshaped = k.reshape(-1, 3)
        
        value = np.array([self.calculate_fan(temperature, g_i, k_i, n_g, n_q, eta) for g_i, k_i in zip(g_reshaped, k_reshaped)])
        
        return value.reshape(k[..., 0].shape)
