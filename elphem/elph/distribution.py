import sys
import numpy as np

from elphem.const.unit import Energy

# Setting up system-related constants and warning filters
float_min = sys.float_info.min
float_max = sys.float_info.max

def safe_divide(a: np.ndarray | float | int, b: np.ndarray | float | int, default=np.nan):
    """
    Safely divides two numbers, arrays, or a combination thereof, with optional handling of division by zero.

    Args:
        a (np.ndarray | float | int): Numerator, can be a number or an array.
        b (np.ndarray | float | int): Denominator, can be a number or an array.
        default (float | int, optional): Value to use when division by zero occurs. Default is NaN.

    Returns:
        np.ndarray: Result of the division, with division by zero handled gracefully.
    """
    a_array = np.full_like(b, a)
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.divide(a_array, b, out=np.full_like(b, default), where=b != 0)
    return result

def boltzmann_distribution(temperature: float, energy: float | np.ndarray) -> float | np.ndarray:
    """
    Calculates the occupation number of particles following the Boltzmann distribution.

    Args:
        temperature (float): Temperature in Kelvin.
        energy (float | np.ndarray): Energy value(s) in Hartree atomic units.

    Returns:
        float | np.ndarray: Occupation number(s) based on the Boltzmann distribution.
    """
    kbt = max(temperature * Energy.KELVIN["->"], float_min)
    beta = safe_divide(1.0, kbt, default=float_max)

    ln = - beta * energy
    return np.exp(ln, out=np.zeros_like(energy), where=ln > -np.log(float_max))

def fermi_distribution(temperature: float, energy: float | np.ndarray) -> float | np.ndarray:
    """
    Calculates the occupation number of particles following the Fermi-Dirac distribution.

    Args:
        temperature (float): Temperature in Kelvin.
        energy (float | np.ndarray): Energy value(s) in Hartree atomic units.

    Returns:
        float | np.ndarray: Occupation number(s) based on the Fermi-Dirac distribution.
    """
    boltzmann_factor = boltzmann_distribution(temperature, energy)
    inv_boltzmann_factor = safe_divide(1.0, boltzmann_factor)
    return safe_divide(1.0, inv_boltzmann_factor + 1.0)

def bose_distribution(temperature: float, energy: float | np.ndarray) -> float | np.ndarray:
    """
    Calculates the occupation number of particles following the Bose-Einstein distribution.

    Args:
        temperature (float): Temperature in Kelvin.
        energy (float | np.ndarray): Energy value(s) in Hartree atomic units.

    Returns:
        float | np.ndarray: Occupation number(s) based on the Bose-Einstein distribution.
    """
    boltzmann_factor = boltzmann_distribution(temperature, energy)
    inv_boltzmann_factor = safe_divide(1.0, boltzmann_factor)
    return safe_divide(1.0, inv_boltzmann_factor - 1.0)

def gaussian_distribution(sigma: float, energy: float | np.ndarray) -> float | np.ndarray:
    """
    Calculates the probability density of particles following the Gaussian distribution.

    Args:
        sigma (float): Standard deviation of the Gaussian distribution.
        energy (float | np.ndarray): Energy value(s) to evaluate the Gaussian function.

    Returns:
        float | np.ndarray: Probability density(s) based on the Gaussian distribution.
    """
    if sigma == 0.0:
        raise ValueError("Sigma must not be zero.")
    return np.exp(- energy ** 2 / (2.0 * sigma ** 2)) / (np.sqrt(2.0 * np.pi) * sigma)