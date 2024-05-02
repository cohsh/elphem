import numpy as np

from elphem.common.function import safe_divide
from elphem.electron.free import FreeElectron
from elphem.phonon.debye import DebyePhonon

class GreenFunction:
    def __init__(self, electron: FreeElectron, phonon: DebyePhonon, sigma: float = 0.00001, eta: float = 0.0001):
        self.sigma = sigma
        self.eta = eta
        self.gaussian_coefficient_a = 2.0 * self.sigma ** 2
        self.gaussian_coefficient_b = np.sqrt(2.0 * np.pi) * self.sigma

        self.poles = np.array([electron.eigenenergies + phonon.eigenenergies, electron.eigenenergies - phonon.eigenenergies])
        
        self.weights = np.array([electron.occupations + phonon.occupations, 1.0 - electron.occupations + phonon.occupations])
    
    def calculate(self, omega: float) -> np.ndarray:
        omega_minus_poles = omega - self.poles
        return self.calculate_real_part(omega_minus_poles) + 1.0j * self.calculate_imag_part(omega_minus_poles)
    
    def calculate_real_part(self, omega_minus_poles: np.ndarray) -> np.ndarray:
        real_part = np.nansum(safe_divide(self.weights, omega_minus_poles + self.eta).real, axis=0)
        
        return real_part
    
    def calculate_imag_part(self, omega_minus_poles: np.ndarray) -> np.ndarray:
        imag_part = np.nansum(np.exp(- self.weights * omega_minus_poles ** 2 / self.gaussian_coefficient_a), axis=0) / self.gaussian_coefficient_b * np.pi
        
        return imag_part