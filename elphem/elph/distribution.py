import sys
import numpy as np
import warnings

from elphem.const.unit import Energy

float_min = sys.float_info.min
float_max = sys.float_info.max
ln_max = np.log(float_max)

warnings.simplefilter("ignore", RuntimeWarning)

def boltzmann_distribution(temperature: float, energy: float | np.ndarray) -> float | np.ndarray:
    # Calculates the occupation number of particles that follow the Blotzmann distribution.

    kbt = temperature * Energy.KELVIN["->"]
    n = 0.0

    if kbt > float_min:
        beta = 1.0 / kbt  # inverse temperature
    else:
        beta = np.sqrt(float_max)
    ln = - beta * energy
    n = np.exp(ln)
    return n

def fermi_distribution(temperature: float, energy: float | np.ndarray) -> float | np.ndarray:
    # Calculates the occupation number of particles that follow the Fermi-Dirac distribution.

    boltzmann_factor = boltzmann_distribution(temperature, energy)
    n = 1.0 / (1.0 / boltzmann_factor + 1.0)
    return n

def bose_distribution(temperature: float, energy: float | np.ndarray) -> float | np.ndarray:
    # Calculates the occupation number of particles that follow the Bose-Einstein distribution.

    boltzmann_factor = boltzmann_distribution(temperature, energy)
    n = 1.0 / (1.0 / boltzmann_factor - 1.0)
    return n

def gaussian_distribution(sigma: float, energy: float | np.ndarray) -> float | np.ndarray:
    # Calculates the occupation number of particles that follow the Gaussian distribution.

    n = np.exp(- energy ** 2 / (2.0 * sigma ** 2)) / (np.sqrt(2.0 * np.pi) * sigma)
    return n