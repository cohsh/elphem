import numpy as np
from dataclasses import dataclass

from elphem.common.unit import Byte
from elphem.common.stdout import ProgressBar
from elphem.common.function import safe_divide
from elphem.electron.free import FreeElectron
from elphem.phonon.debye import DebyePhonon
from elphem.elph.distribution import fermi_distribution, bose_distribution

@dataclass
class ElectronPhonon:
    """Calculate the electron-phonon components for electronic states in a lattice.

    Attributes:
        electron (FreeElectron): Free electron model for the lattice.
        phonon (DebyeModel): Phonon model using Debye approximation.
        temperature (float): Temperature of the system in Kelvin.
        sigma (float): Smearing parameter for the Gaussian distribution, defaults to 0.01.
        eta (float): Small positive constant to ensure numerical stability, defaults to 0.01.
        effective_potential (float): Effective potential used in electron-phonon coupling calculation, defaults to 1.0 / 16.0.
    """

    electron: FreeElectron
    phonon: DebyePhonon
    temperature: float
    n_qs: np.ndarray
    sigma: float = 0.0001
    eta: float = 0.1
    effective_potential: float = 1.0 / 16.0

    def __post_init__(self):
        self.coefficient = 1.0 / np.prod(self.n_qs)

        self.gaussian_coefficient_a = 2.0 * self.sigma ** 2
        self.gaussian_coefficient_b = np.sqrt(2.0 * np.pi) * self.sigma

    def get_coupling(self, g1: np.ndarray, g2: np.ndarray, q: np.ndarray) -> np.ndarray:
        """Calculate the lowest-order electron-phonon coupling between states.

        Args:
            g1 (np.ndarray): Initial G-vector in reciprocal space.
            g2 (np.ndarray): Final G-vector in reciprocal space.
            q (np.ndarray): Phonon wave vector in reciprocal space.

        Returns:
            np.ndarray: The electron-phonon coupling strength for the given vectors.
        """
        
        phonon_eigenenergy = self.phonon.get_eigenenergy(q)
        phonon_eigenvector = self.phonon.get_eigenvector(q)
        zero_point_length = safe_divide(1.0, np.sqrt(2.0 * self.phonon.lattice.mass * phonon_eigenenergy))
        
        coupling = -1.0j * self.effective_potential * np.sum((q + g1 - g2) * phonon_eigenvector, axis=-1) * zero_point_length
        
        return coupling

    def get_ggkq_grid(self, k_array: np.ndarray) -> tuple:
        q_array = self.electron.lattice.reciprocal_cell.get_monkhorst_pack_grid(*self.n_qs)

        n_band = self.electron.n_band
        n_k = len(k_array)
        n_q = len(q_array)

        shape = (n_band, n_band, n_k, n_q, 3)
        
        g1 = np.broadcast_to(self.electron.reciprocal_vectors[:, np.newaxis, np.newaxis, np.newaxis, :], shape)
        g2 = np.broadcast_to(self.electron.reciprocal_vectors[np.newaxis, :, np.newaxis, np.newaxis, :], shape)
        k = np.broadcast_to(k_array[np.newaxis, np.newaxis, :, np.newaxis, :], shape)
        q = np.broadcast_to(q_array[np.newaxis, np.newaxis, np.newaxis, :, :], shape)
        
        return g1, g2, k, q

    def get_omega_independent_values(self, k_array: np.ndarray) -> tuple:
        g1, g2, k, q = self.get_ggkq_grid(k_array)
        
        electron_eigenenergy_inter = self.electron.get_eigenenergy(k + q + g2)
        phonon_eigenenergy = self.phonon.get_eigenenergy(q)

        fermi = fermi_distribution(self.temperature, electron_eigenenergy_inter)
        bose = bose_distribution(self.temperature, phonon_eigenenergy)

        occupations = {}

        occupations['+'] = fermi + bose
        occupations['-'] = 1.0 - fermi + bose

        coupling2 = np.abs(self.get_coupling(g1, g2, q)) ** 2

        return electron_eigenenergy_inter, phonon_eigenenergy, occupations, coupling2

    def get_green_function(self, omega: float, electron_eigenenergy: np.ndarray, phonon_eigenenergy: np.ndarray, occupations: dict) -> np.ndarray:
        denominators = {}

        denominators['-'] = omega - electron_eigenenergy - phonon_eigenenergy
        denominators['+'] = omega - electron_eigenenergy + phonon_eigenenergy

        green_function = np.zeros(electron_eigenenergy.shape, dtype='complex')

        for sign in denominators.keys():
            green_function += occupations[sign] * self.get_green_function_real(denominators[sign])
            green_function += 1.0j * np.pi * occupations[sign] * self.get_green_function_imag(denominators[sign])
        
        return green_function

    def get_self_energy(self, omega: float, k_array: np.ndarray) -> np.ndarray:
        """Calculate a single value of Fan self-energy for given wave vectors.

        Args:
            omega (float): a frequency.
            k_array (np.ndarray): k-vectors.

        Returns:
            complex: The Fan self-energy term as a complex number.
        """
        
        electron_eigenenergy, phonon_eigenenergy, occupations, coupling2 = self.get_omega_independent_values(k_array)
        
        green_function = self.get_green_function(omega, electron_eigenenergy, phonon_eigenenergy, occupations)

        self_energy = np.nansum(coupling2 * green_function, axis=(1, 3)) * self.coefficient
        
        return self_energy

    def get_spectrum(self, k_names: list[str], n_split: int, n_omega: int, range_omega: list[float]) -> tuple:
        """
        Calculate the spectral function along a specified path in the Brillouin zone.

        Args:
            k_names (list[str]): List of special points defining the path through the Brillouin zone.
            n_split (int): Number of points between each special point.
            n_q (np.ndarray): A numpy array specifying the density of q-grid points in each direction.
            n_omega (int): Number of points in the energy range.
            range_omega (list[float]): The range of energy values over which to calculate the spectrum.

        Returns:
            tuple: A tuple containing the path x-coordinates, energy values, the calculated spectrum, and x-coordinates of special points.
        """
        
        g = self.electron.reciprocal_vectors
        
        k_path = self.electron.lattice.reciprocal_cell.get_path(k_names, n_split)
        eig = np.array([self.electron.get_eigenenergy(k_path.values + g_i) for g_i in g])

        omega_array = np.linspace(range_omega[0], range_omega[1], n_omega)
        
        shape = (len(k_path.values), n_omega)
        spectrum = np.empty(shape)
        
        electron_eigenenergy, phonon_eigenenergy, occupations, coupling2 = self.get_omega_independent_values(k_path.values)

        count = 0

        progress_bar = ProgressBar('Spectrum', n_omega)
        for omega in omega_array:
            green_function = self.get_green_function(omega, electron_eigenenergy, phonon_eigenenergy, occupations)
            
            self_energy = np.nansum(coupling2 * green_function, axis=(1, 3)) * self.coefficient

            numerator = - self_energy.imag / np.pi
            denominator = (omega - eig - self_energy.real) ** 2 + self_energy.imag ** 2

            fraction = safe_divide(numerator, denominator)
            
            spectrum[..., count] = np.nansum(fraction, axis=0)
            
            count += 1

            progress_bar.print(count)

        return k_path.distances, omega_array, spectrum, k_path.special_distances

    def get_green_function_real(self, omega: np.ndarray) -> np.ndarray:
        green_function_real = safe_divide(1.0, omega + self.eta * 1.0j).real
        
        return green_function_real
    
    def get_green_function_imag(self, omega: np.ndarray) -> np.ndarray:
        green_function_imag = np.exp(- omega ** 2 / self.gaussian_coefficient_a) / self.gaussian_coefficient_b
        
        return green_function_imag