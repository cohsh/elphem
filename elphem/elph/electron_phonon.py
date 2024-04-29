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
    sigma: float = 0.00001
    eta: float = 0.1
    effective_potential: float = 1.0 / 16.0

    def __post_init__(self):
        self.coefficient = 1.0 / self.phonon.n_q

        self.gaussian_coefficient_a = 2.0 * self.sigma ** 2
        self.gaussian_coefficient_b = np.sqrt(2.0 * np.pi) * self.sigma
        
        self._set_omega_independent_values()

    def get_coupling(self, g1_array: np.ndarray, g2_array: np.ndarray, q_array: np.ndarray) -> np.ndarray:
        """Calculate the lowest-order electron-phonon coupling between states.

        Args:
            g1 (np.ndarray): Initial G-vector in reciprocal space.
            g2 (np.ndarray): Final G-vector in reciprocal space.
            q (np.ndarray): Phonon wave vector in reciprocal space.

        Returns:
            np.ndarray: The electron-phonon coupling strength for the given vectors.
        """
        
        phonon_eigenenergies = self.phonon.get_eigenenergies(q_array)
        phonon_eigenvectors = self.phonon.get_eigenvectors(q_array)
        zero_point_lengths = safe_divide(1.0, np.sqrt(2.0 * self.lattice.mass * phonon_eigenenergies))

        coupling = -1.0j * self.effective_potential * np.sum((q_array + g1_array - g2_array) * phonon_eigenvectors, axis=-1) * zero_point_lengths

        return coupling

    def get_ggkq_grid(self) -> tuple:
        shape = (self.electron.n_band, self.electron.n_band, self.electron.n_k, self.phonon.n_q, 3)
        
        g1 = np.broadcast_to(self.electron.g[:, np.newaxis, np.newaxis, np.newaxis, :], shape)
        g2 = np.broadcast_to(self.electron.g[np.newaxis, :, np.newaxis, np.newaxis, :], shape)
        k = np.broadcast_to(self.electron.k[np.newaxis, np.newaxis, :, np.newaxis, :], shape)
        q = np.broadcast_to(self.phonon.q[np.newaxis, np.newaxis, np.newaxis, :, :], shape)
        
        return g1, g2, k, q

    def _set_omega_independent_values(self) -> None:
        g1, g2, k, q = self.get_ggkq_grid()

        self.electron_inter = self.electron.derive(g2 + k + q)

        fermi = fermi_distribution(self.temperature, self.electron_inter.eigenenergies)
        bose = bose_distribution(self.temperature, self.phonon.eigenenergies)

        occupations = {}

        occupations['+'] = fermi + bose
        occupations['-'] = 1.0 - fermi + bose

        coupling2 = np.abs(self.get_coupling(g1, g2, q)) ** 2

        return electron_eigenenergies, occupations, coupling2

    def get_self_energy(self, omega: float) -> np.ndarray:
        """Calculate a single value of Fan self-energy for given wave vectors.

        Args:
            omega (float): a frequency.
            k_array (np.ndarray): k-vectors.

        Returns:
            complex: The Fan self-energy term as a complex number.
        """
        
        electron_eigenenergies, occupations, coupling2 = self.get_omega_independent_values()
        
        green_function = self.get_green_function(omega, electron_eigenenergies, occupations)

        self_energy = np.nansum(coupling2 * green_function, axis=(1, 3)) * self.coefficient
        
        return self_energy

    def get_spectrum_with_path(self, k_names: list[str], n_split: int, n_omega: int, range_omega: list[float]) -> tuple:
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
        eig = self.electron.get_eigenenergies(k_path.values, g)

        omega_array = np.linspace(range_omega[0], range_omega[1], n_omega)
        
        shape = (len(k_path.values), n_omega)
        spectrum = np.empty(shape)
        
        electron_eigenenergies, occupations, coupling2 = self.get_omega_independent_values(k_path.values)

        count = 0

        progress_bar = ProgressBar('Spectrum', n_omega)
        for omega in omega_array:
            green_function = self.get_green_function(omega, electron_eigenenergies, occupations)
            
            self_energy = np.nansum(coupling2 * green_function, axis=(1, 3)) * self.coefficient

            numerator = - self_energy.imag / np.pi
            denominator = (omega - eig - self_energy.real) ** 2 + self_energy.imag ** 2

            fraction = safe_divide(numerator, denominator)
            
            spectrum[..., count] = np.nansum(fraction, axis=0)
            
            count += 1

            progress_bar.print(count)

        return k_path.derive(spectrum), omega_array

    def get_green_function(self, omega: float, electron_eigenenergy: np.ndarray, occupations: dict) -> np.ndarray:
        denominators = {}

        denominators['-'] = omega - electron_eigenenergy - self.phonon.eigenenergies
        denominators['+'] = omega - electron_eigenenergy + self.phonon.eigenenergies

        green_function = np.zeros(electron_eigenenergy.shape, dtype='complex')

        for sign in denominators.keys():
            green_function += occupations[sign] * self.get_green_function_real(denominators[sign])
            green_function += 1.0j * np.pi * occupations[sign] * self.get_green_function_imag(denominators[sign])
        
        return green_function

    def get_green_function_real(self, omega: np.ndarray) -> np.ndarray:
        green_function_real = safe_divide(1.0, omega + self.eta * 1.0j).real
        
        return green_function_real
    
    def get_green_function_imag(self, omega: np.ndarray) -> np.ndarray:
        green_function_imag = np.exp(- omega ** 2 / self.gaussian_coefficient_a) / self.gaussian_coefficient_b
        
        return green_function_imag