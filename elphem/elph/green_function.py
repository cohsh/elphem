import numpy as np

from elphem.common.function import safe_divide
from elphem.common.distribution import fermi_distribution, bose_distribution
from elphem.electron.electron import Electron
from elphem.phonon.phonon import Phonon

class GreenFunction:
    """Class for Green functions
        Attributes:
            eta (float): Convergence factor for calculating real parts
            gaussian_coefficient_a (float): Coefficient for calculating imaginary parts (a)
            gaussian_coefficient_b (float): Coefficient for calculating imaginary parts (b)
            poles (np.ndarray): A numpy array of poles
            weights (np.ndarray): A numpy array of weights for each pole
    """
    def __init__(self, electron: Electron, phonon: Phonon, temperature: float, sigma: float, eta: float):
        # set attributes
        self.eta = eta
        self.gaussian_coefficient_a = 2.0 * sigma ** 2
        self.gaussian_coefficient_b = np.sqrt(2.0 * np.pi) * sigma
        self.poles = np.array([
            electron.eigenenergies + phonon.eigenenergies,
            electron.eigenenergies - phonon.eigenenergies
            ])
        
        # prepare occupations
        electron_occupations = fermi_distribution(temperature, electron.eigenenergies)
        phonon_occupations = bose_distribution(temperature, phonon.eigenenergies)

        # set weights
        self.weights = np.array([
            1.0 - electron_occupations + phonon_occupations,
            electron_occupations + phonon_occupations
            ])
    
    def calculate(self, omega: float) -> np.ndarray:
        """Calculate Green function

        Args:
            omega (float): A frequency

        Returns:
            np.ndarray: A numpy array of Green function (complex)
        """
        omega_minus_poles = omega - self.poles
        
        return self.calculate_real_part(omega_minus_poles) + 1.0j * self.calculate_imag_part(omega_minus_poles)
    
    def calculate_real_part(self, omega_minus_poles: np.ndarray) -> np.ndarray:
        """Calculate real part of Green function

        Args:
            omega_minus_poles (np.ndarray): A numpy array of frequencies

        Returns:
            np.ndarray: A numpy array of real part of Green function
        """
        real_part = np.nansum(safe_divide(self.weights, omega_minus_poles + 1.0j * self.eta).real, axis=0)
        
        return real_part
    
    def calculate_imag_part(self, omega_minus_poles: np.ndarray) -> np.ndarray:
        """Calculate imaginary part of Green function

        Args:
            omega_minus_poles (np.ndarray): A numpy array of frequencies

        Returns:
            np.ndarray: A numpy array of imaginary part of Green function
        """
        imag_part = - np.nansum(self.weights * np.exp(- omega_minus_poles ** 2 / self.gaussian_coefficient_a), axis=0) / self.gaussian_coefficient_b * np.pi
        
        return imag_part