import numpy as np
from dataclasses import dataclass

from elphem.lattice.empty import EmptyLattice
from elphem.electron.free import FreeElectron
from elphem.phonon.debye import DebyeModel
from elphem.elph.coupling import Coupling
from elphem.elph.distribution import fermi_distribution, bose_distribution

@dataclass
class Strength:
    lattice: EmptyLattice
    electron: FreeElectron
    phonon: DebyeModel
    sigma: float = 0.01

@dataclass
class Strength2nd(Strength):
    def validate_inputs(self, temperature, eta):
        if not isinstance(temperature, (int, float)) or temperature < 0:
            raise ValueError("Temperature must be a not-negative number.")
        if not isinstance(eta, (int, float)):
            raise ValueError("eta must be a number.")

    def calculate(self, temperature: float, g: np.ndarray, k: np.ndarray,
                    n_g: np.ndarray, n_q: np.ndarray, eta=0.01) -> np.ndarray:
        """
        Calculate 2nd-order coupling strengths.
        
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
                        n_g: np.ndarray, n_q: np.ndarray, eta=0.01) -> float:
        """
        Calculate single values of coupling strengths.
        
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

        g = Coupling.first_order(g1, g2, q, self.phonon)
        
        delta_energy = electron_energy_nk - electron_energy_mkq
        # Real Part
        green_part_real = - ((1.0 - fermi + bose) / (delta_energy - omega + eta * 1.0j) ** 2
                           + (fermi + bose) / (delta_energy + omega + eta * 1.0j) ** 2)

        diff_selfen_real = - np.nansum(np.abs(g) ** 2 * green_part_real).real
        
        coeff = 2.0 * np.pi / np.prod(n_q)
        
        return diff_selfen_real * coeff