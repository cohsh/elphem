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
    temperature: float
    sigma: float = 0.01
    eta: float = 0.01
    effective_potential: float = 1.0 / 16.0
    
    def coupling(self, g1: np.ndarray, g2: np.ndarray, q: np.ndarray) -> np.ndarray:
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

    def calculate_fan_term(self, g: np.ndarray, k: np.ndarray, n_q: np.ndarray) -> complex:
        """
        Calculate single values of Fan self-energy.
        
        Args
            g: A numpy array representing a single G-vector
            k: A numpy array representing a single k-vector
            n_g: A numpy array representing the dense of intermediate G-vectors
            n_q: A numpy array representing the dense of intermediate q-vectors
            eta: A value of the convergence factor. The default value is 0.01 Hartree.
            
        Return
            A complex value representing Fan self-energy.
        """
            
        g_inter, q = self.electron.grid(n_q) # Generate intermediate G, q grid.

        omega = self.phonon.eigenenergy(q)
        bose = bose_distribution(self.temperature, omega)
        
        coeff = 2.0 * np.pi / np.prod(n_q)

        epsilon = self.electron.eigenenergy(k)
        epsilon_inter = self.electron.eigenenergy(k + q)

        fermi = fermi_distribution(self.temperature, epsilon_inter)

        coupling = self.coupling(g, g_inter, q)
    
        delta_energy = epsilon - epsilon_inter
        # Real Part
        green_part_real = ((1.0 - fermi + bose) / (delta_energy - omega + self.eta * 1.0j)
                        + (fermi + bose) / (delta_energy + omega + self.eta * 1.0j)).real

        # Imaginary Part
        green_part_imag = ((1.0 - fermi + bose) * gaussian_distribution(self.sigma, delta_energy - omega)
                        + (fermi + bose) * gaussian_distribution(self.sigma, delta_energy + omega))

        selfen = (np.nansum(np.abs(coupling) ** 2 * green_part_real) 
                        + 1.0j * np.nansum(np.abs(coupling) ** 2 * green_part_imag))
        
        return selfen * coeff

    def calculate_coupling_strength(self, g: np.ndarray, k: np.ndarray, n_q: np.ndarray) -> float:
        """
        Calculate electron-phonon coupling strengths.
        
        Args
        """
        
        g_inter, q = self.electron.grid(n_q) # Generate intermediate G, q grid.

        omega = self.phonon.eigenenergy(q)
        bose = bose_distribution(self.temperature, omega)
        
        coeff = 2.0 * np.pi / np.prod(n_q)

        epsilon = self.electron.eigenenergy(k)
        epsilon_inter = self.electron.eigenenergy(k + q)

        fermi = fermi_distribution(self.temperature, epsilon_inter)

        coupling = self.coupling(g, g_inter, q)
    
        delta_energy = epsilon - epsilon_inter
        # Real Part
        partial_green_part_real = - ((1.0 - fermi + bose) / (delta_energy - omega + self.eta * 1.0j) ** 2
                        + (fermi + bose) / (delta_energy + omega + self.eta * 1.0j) ** 2).real

        coupling_strength = - np.nansum(np.abs(coupling) ** 2 * partial_green_part_real)
        
        return coupling_strength * coeff
    
    def calculate_qp_strength(self, g: np.ndarray, k: np.ndarray, n_q: np.ndarray) -> float:
        """
        Calculate quasiparticle strengths
        
        Args
        """
        coupling_strength = self.calculate_coupling_strength(g, k, n_q)
        qp_strength = safe_divide(1.0, 1.0 + coupling_strength)

        return qp_strength