import numpy as np

from elphem.common.function import safe_divide
from elphem.common.distribution import fermi_distribution, bose_distribution
from elphem.electron.free import FreeElectron
from elphem.phonon.debye import DebyePhonon

class GreenFunction:
    def __init__(self, electron: FreeElectron, phonon: DebyePhonon, sigma: float, eta: float):
        self.electron = electron
        self.phonon = phonon
        
        self.sigma = sigma
        self.eta = eta
        self.gaussian_coefficient_a = 2.0 * self.sigma ** 2
        self.gaussian_coefficient_b = np.sqrt(2.0 * np.pi) * self.sigma

        self.poles = np.array([
            electron.eigenenergies + phonon.eigenenergies,
            electron.eigenenergies - phonon.eigenenergies
            ])
        
        self.weights = np.zeros((2,))
    
    def calculate(self, temperature: float, omega: float) -> np.ndarray:
        self.update_weights(temperature)
        omega_minus_poles = omega - self.poles
        
        return self._calculate_real_part(omega_minus_poles) + 1.0j * self._calculate_imag_part(omega_minus_poles)
    
    def _calculate_real_part(self, omega_minus_poles: np.ndarray) -> np.ndarray:
        real_part = np.nansum(safe_divide(self.weights, omega_minus_poles + 1.0j * self.eta).real, axis=0)
        
        return real_part
    
    def _calculate_imag_part(self, omega_minus_poles: np.ndarray) -> np.ndarray:
        imag_part = - np.nansum(self.weights * np.exp(- omega_minus_poles ** 2 / self.gaussian_coefficient_a), axis=0) / self.gaussian_coefficient_b * np.pi
        
        return imag_part
    
    def update_weights(self, temperature: float) -> None:
        electron_occupations = fermi_distribution(temperature, self.electron.eigenenergies)
        phonon_occupations = bose_distribution(temperature, self.phonon.eigenenergies)
        self.weights = np.array([
            1.0 - electron_occupations + phonon_occupations,
            electron_occupations + phonon_occupations
            ])