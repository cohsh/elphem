import numpy as np
from dataclasses import dataclass

from elphem.const.unit import *
from elphem.lattice.empty import EmptyLattice
from elphem.electron.free import FreeElectron
from elphem.phonon.debye import DebyeModel
from elphem.elph.distribution import fermi_distribution, bose_distribution, gaussian_distribution

@dataclass
class SelfEnergy:
    lattice: EmptyLattice
    electron: FreeElectron
    phonon: DebyeModel
    sigma: float = 0.01
    effective_potential: float = 1 / 16

@dataclass
class SelfEnergy2nd(SelfEnergy):
    def calculate(self, temperature: float, g: np.ndarray, k: np.ndarray,
                    n_g: np.ndarray, n_q: np.ndarray, eta=0.01) -> np.ndarray:
        shape_return = k[..., 0].shape
        
        g_reshaped = g.reshape(int(g.size / 3), 3)
        k_reshaped = k.reshape(int(k.size / 3), 3)

        value = np.empty((len(k_reshaped),), dtype=complex)

        for i in range(len(k_reshaped)):
            value[i] = self.calculate_fan(temperature, g_reshaped[i], k_reshaped[i], n_g, n_q, eta=eta)
        
        value = value.reshape(shape_return)

        return value
    
    def calculate_fan(self, temperature: float, g1: np.ndarray, k: np.ndarray, 
                        n_g: np.ndarray, n_q: np.ndarray, eta=0.01) -> np.ndarray:
        g2, q = self.electron.grid(n_g, n_q)

        electron_energy_nk = self.electron.eigenenergy(k + g1)
        electron_energy_mkq = self.electron.eigenenergy(k + g2 + q)

        omega = self.phonon.eigenenergy(q)

        bose = bose_distribution(temperature, omega)
        fermi = fermi_distribution(temperature, electron_energy_mkq)

        g = self.coupling(g1, g2, k, q)
        
        # Real Part
        green_part_real = (1.0 - fermi + bose) / (electron_energy_nk - electron_energy_mkq - omega + eta * 1.0j)
        green_part_real += (fermi + bose) / (electron_energy_nk - electron_energy_mkq + omega + eta * 1.0j)        

        # Imaginary Part
        green_part_imag = (1.0 - fermi + bose) * gaussian_distribution(self.sigma, electron_energy_nk - electron_energy_mkq - omega)
        green_part_imag += (fermi + bose) * gaussian_distribution(self.sigma, electron_energy_nk - electron_energy_mkq + omega)

        selfen_real = np.nansum(np.abs(g) ** 2 * green_part_real).real
        selfen_imag = np.nansum(np.abs(g) ** 2 * green_part_imag)

        selfen_real *= 2.0 * np.pi / (n_q.prod())
        selfen_imag *= 2.0 * np.pi / (n_q.prod())
        
        return selfen_real + 1.0j * selfen_imag
    
    def coupling(self, g1: np.ndarray, g2: np.ndarray, k: np.ndarray, q: np.ndarray) -> np.ndarray:
        q_norm = np.linalg.norm(q, axis=q.ndim-1)
        q_dot = q_norm ** 2

        delta_g = g1 - g2

        q_dot = 0.0
        for i in range(3):
            q_dot += q[..., i] * delta_g[..., i]

        coupling = 1.0 / np.sqrt(2.0 * self.phonon.mass * self.phonon.speed) * (q_norm ** (-3.0 / 2.0)) * q_dot * self.effective_potential
        
        return coupling