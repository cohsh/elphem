import numpy as np
from dataclasses import dataclass

from elphem.lattice.empty import EmptyLattice
from elphem.electron.free import FreeElectron
from elphem.phonon.debye import DebyeModel
from elphem.elph.distribution import fermi_distribution, bose_distribution, gaussian_distribution, safe_divide

@dataclass
class SelfEnergy:
    lattice: EmptyLattice
    electron: FreeElectron
    phonon: DebyeModel
    sigma: float = 0.01
    effective_potential: float = 1 / 16

@dataclass
class SelfEnergy2nd(SelfEnergy):
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
    
    def calculate_fan(self, temperature: float, g1: np.ndarray, k: np.ndarray, 
                        n_g: np.ndarray, n_q: np.ndarray, eta=0.01) -> complex:
        """
        Calculate single values of Fan self-energy.
        
        Args
            temperature: A temperature in Kelvin.
            g1: A numpy array representing G-vector
            k: A numpy array representing k-vector
            n_g: A numpy array representing the dense of intermediate G-vectors
            n_q: A numpy array representing the dense of intermediate q-vectors
            eta: A value of the convergence factor. The default value is 0.01 Hartree.
            
        Return
            A complex value representing Fan self-energy.
        """
        g2, q = self.electron.grid(n_g, n_q) # Generate intermediate G, q grid.

        electron_energy_nk = self.electron.eigenenergy(k + g1)
        electron_energy_mkq = self.electron.eigenenergy(k + g2 + q)

        omega = self.phonon.eigenenergy(q)
        bose = bose_distribution(temperature, omega)
        fermi = fermi_distribution(temperature, electron_energy_mkq)

        g = self.coupling(g1, g2, k, q)
        
        delta_energy = electron_energy_nk - electron_energy_mkq
        # Real Part
        green_part_real = ((1.0 - fermi + bose) / (delta_energy - omega + eta * 1.0j)
                           + (fermi + bose) / (delta_energy + omega + eta * 1.0j))

        # Imaginary Part
        green_part_imag = ((1.0 - fermi + bose) * gaussian_distribution(self.sigma, delta_energy - omega)
                           + (fermi + bose) * gaussian_distribution(self.sigma, delta_energy + omega))

        selfen_real = np.nansum(np.abs(g) ** 2 * green_part_real).real
        selfen_imag = np.nansum(np.abs(g) ** 2 * green_part_imag)
        
        coeff = 2.0 * np.pi / np.prod(n_q)
        
        return (selfen_real + 1.0j * selfen_imag) * coeff
    
    def coupling(self, g1: np.ndarray, g2: np.ndarray, k: np.ndarray, q: np.ndarray) -> np.ndarray:
        """
        Calculate lowest-order electron-phonon couplings.
        
        Args
            g1, g2: A numpy array representing G-vector
            k: A numpy array representing k-vector
            q: A numpy array representing k-vector
        
        Return
            A value of the elctron-phonon coupling.
        """
        q_norm = np.linalg.norm(q, axis=-1)
        delta_g = g1 - g2
        q_dot = np.sum(q * delta_g, axis=-1) 

        mask = q_norm > 0
        result = np.zeros_like(q_norm)
        
        denominator = np.sqrt(2.0 * self.phonon.mass * self.phonon.speed) * q_norm ** 1.5
        result[mask] = safe_divide(self.effective_potential * q_dot[mask], denominator[mask])
        
        return result