import numpy as np

from elphem.common.unit import Energy

np.seterr(divide='ignore', over='ignore')

def boltzmann_distribution(temperature: float, energy: float | np.ndarray) -> float | np.ndarray:
    """
    Calculates the occupation number of particles following the Boltzmann distribution.

    Args:
        temperature (float): Temperature in Kelvin.
        energy (float | np.ndarray): Energy value(s) in Hartree atomic units.

    Returns:
        float | np.ndarray: Occupation number(s) based on the Boltzmann distribution.
    """

    if temperature != 0.0:
        beta = 1.0 / (temperature * Energy.KELVIN["->"])
        exponent = - beta * energy
        exponent = np.clip(exponent, -np.inf, 700.0)
        return np.exp(exponent)
    else:
        return np.where(energy > 0.0, 0.0, np.inf)

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
    inv_boltzmann_factor = 1.0 / boltzmann_factor
    
    return 1.0 / (inv_boltzmann_factor + 1.0)

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
    inv_boltzmann_factor = 1.0 / boltzmann_factor
    
    return 1.0 / (inv_boltzmann_factor - 1.0)

def gaussian_distribution(sigma: float, energy: float | np.ndarray) -> float | np.ndarray:
    """
    Calculates the probability density of particles following the Gaussian distribution.

    Args:
        sigma (float): Standard deviation of the Gaussian distribution.
        energy (float | np.ndarray): Energy value(s) to evaluate the Gaussian function.

    Returns:
        float | np.ndarray: Probability density(s) based on the Gaussian distribution.
    """
    
    return np.exp(- energy ** 2 / (2.0 * sigma ** 2)) / (np.sqrt(2.0 * np.pi) * sigma)