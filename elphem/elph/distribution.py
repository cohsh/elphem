import sys
import numpy as np
import warnings

from elphem.const.unit import Energy

# Setting up system-related constants and warning filters
float_min = sys.float_info.min
float_max = sys.float_info.max
warnings.simplefilter("ignore", RuntimeWarning)

def boltzmann_distribution(temperature: float, energy: float | np.ndarray) -> float | np.ndarray:
    """
    Calculates the occupation number of particles that follow the Boltzmann distribution.

    Args:
        temperature: Temperature in Kelvin
        energy: Energy value(s) in Hartree atomic units

    Returns:
        Occupation number based on the Boltzmann distribution
    """
    
    kbt = temperature * Energy.KELVIN["->"]

    if kbt > float_min:
        beta = 1.0 / kbt  # inverse temperature
    else:
        beta = np.sqrt(float_max)

    ln = - beta * energy
    return np.exp(ln)

def fermi_distribution(temperature: float, energy: float | np.ndarray) -> float | np.ndarray:
    """
    Calculates the occupation number of particles that follow the Fermi-Dirac distribution.

    Args:
        temperature: Temperature in Kelvin.
        energy: Energy value(s) in Hartree atomic units.

    Returns:
        Occupation number(s) based on the Fermi-Dirac distribution.
    """
    
    boltzmann_factor = boltzmann_distribution(temperature, energy)
    return 1.0 / (1.0 / boltzmann_factor + 1.0)

def bose_distribution(temperature: float, energy: float | np.ndarray) -> float | np.ndarray:
    """
    Calculates the occupation number of particles that follow the Bose-Einstein distribution.

    Args:
        temperature: Temperature in Kelvin.
        energy: Energy value(s) in Hartree atomic units.

    Returns:
        Occupation number(s) based on the Bose-Einstein distribution.
    """
    
    boltzmann_factor = boltzmann_distribution(temperature, energy)
    return 1.0 / (1.0 / boltzmann_factor - 1.0)

def gaussian_distribution(sigma: float, energy: float | np.ndarray) -> float | np.ndarray:
    """
    Calculates the occupation number of particles that follow the Gaussian distribution.

    Args:
        sigma: Standard deviation of the Gaussian distribution.
        energy: Energy value(s) in Hartree atomic units.

    Returns:
        Occupation number(s) based on the Gaussian distribution.
    """
    
    return np.exp(- energy ** 2 / (2.0 * sigma ** 2)) / (np.sqrt(2.0 * np.pi) * sigma)