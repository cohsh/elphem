import sys
import numpy as np
import warnings

from elphem.const.unit import Energy

# Setting up system-related constants and warning filters
float_min = sys.float_info.min
float_max = sys.float_info.max

def safe_divide(a: np.ndarray | float | int, b: np.ndarray | float | int, default=np.nan):
    """Safely divide two numbers, arrays, or a combination thereof."""
    a_array = np.full_like(b, a)
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.divide(a_array, b, out=np.full_like(b, default), where=b != 0)
    return result

def boltzmann_distribution(temperature: float, energy: float | np.ndarray) -> float | np.ndarray:
    """
    Calculates the occupation number of particles that follow the Boltzmann distribution.

    Args:
        temperature: Temperature in Kelvin
        energy: Energy value(s) in Hartree atomic units

    Returns:
        Occupation number based on the Boltzmann distribution
    """
    kbt = max(temperature * Energy.KELVIN["->"], float_min)
    beta = safe_divide(1.0, kbt, default=float_max)

    ln = - beta * energy
    return np.exp(ln, out=np.zeros_like(energy), where=ln > -np.log(float_max))

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
    inv_boltzmann_factor = safe_divide(1.0, boltzmann_factor)
    return safe_divide(1.0, inv_boltzmann_factor + 1.0)

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
    inv_boltzmann_factor = safe_divide(1.0, boltzmann_factor)
    return safe_divide(1.0, inv_boltzmann_factor - 1.0)

def gaussian_distribution(sigma: float, energy: float | np.ndarray) -> float | np.ndarray:
    """
    Calculates the occupation number of particles that follow the Gaussian distribution.

    Args:
        sigma: Standard deviation of the Gaussian distribution.
        energy: Energy value(s) in Hartree atomic units.

    Returns:
        Occupation number(s) based on the Gaussian distribution.
    """
    if sigma == 0:
        raise ValueError("Sigma must not be zero.")
    return np.exp(- energy ** 2 / (2.0 * sigma ** 2)) / (np.sqrt(2.0 * np.pi) * sigma)