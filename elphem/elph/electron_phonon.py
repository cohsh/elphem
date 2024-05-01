import numpy as np
from dataclasses import dataclass

from elphem.common.unit import Byte
from elphem.common.stdout import ProgressBar
from elphem.common.function import safe_divide
from elphem.electron.free import FreeElectron
from elphem.phonon.debye import DebyePhonon
from elphem.elph.green_function import GreenFunction

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
    sigma: float = 0.00001
    eta: float = 0.1
    effective_potential: float = 1.0 / 16.0

    def __post_init__(self):
        g1, g2, k, q = self.create_ggkq_grid()

        self.electron.update(g1, k)
        self.phonon.update(q)
        self.electron_inter = self.electron.clone_with_gk_grid(g2, k + q)
        
        self.green_function = GreenFunction(self.electron, self.phonon)

        self.coupling2 = np.abs(self.get_coupling(g1, g2, q)) ** 2

    def calculate_couplings(self) -> np.ndarray:
        """Calculate the lowest-order electron-phonon coupling between states.

        Args:
            g1 (np.ndarray): Initial G-vector in reciprocal space.
            g2 (np.ndarray): Final G-vector in reciprocal space.
            q (np.ndarray): Phonon wave vector in reciprocal space.

        Returns:
            np.ndarray: The electron-phonon coupling strength for the given vectors.
        """
        
        couplings = -1.0j * self.effective_potential * np.sum((self.phonon.q + self.electron.g1 - self.electron.g2) * self.phonon_eigenvectors, axis=-1) * self.phonon.zero_point_lengths

        return couplings

    def create_ggkq_grid(self) -> tuple:
        shape = (self.electron.n_band, self.electron.n_band, self.electron.n_k, self.phonon.n_q, 3)
        
        g1 = np.broadcast_to(self.electron.g[:, np.newaxis, np.newaxis, np.newaxis, :], shape)
        g2 = np.broadcast_to(self.electron.g[np.newaxis, :, np.newaxis, np.newaxis, :], shape)
        k = np.broadcast_to(self.electron.k[np.newaxis, np.newaxis, :, np.newaxis, :], shape)
        q = np.broadcast_to(self.phonon.q[np.newaxis, np.newaxis, np.newaxis, :, :], shape)
        
        return g1, g2, k, q

    def calculate_self_energies(self, omega: float) -> np.ndarray:
        """Calculate a single value of Fan self-energy for given wave vectors.

        Args:
            omega (float): a frequency.

        Returns:
            complex: The Fan self-energy term as a complex number.
        """
        
        return np.nansum(self.coupling2 * self.green_function.calculate(omega), axis=(1, 3)) * self.coefficient

    def calculate_spectrum(self, omega: float) -> np.ndarray:
        self_energies = self.calculate_self_energies(omega)

        numerator = - self_energies.imag / np.pi
        denominator = (omega - self.electron.eigenenergies - self_energies.real) ** 2 + self_energies.imag ** 2
        
        return np.nansum(safe_divide(numerator, denominator), axis=0)

    def calculate_self_energies_over_range(self, omega_array: np.ndarray | list[float]) -> np.ndarray:
        n_omega = len(omega_array)
        self_energies = np.empty((self.electron.n_k, n_omega))
        
        count = 0
        progress_bar = ProgressBar('Self Energy', n_omega)
        for omega in omega_array:
            self_energies[..., count] = self.calculate_self_energies(omega)
            
            count += 1
            progress_bar.print(count)

        return self_energies
        
    def calculate_spectrum_over_range(self, omega_array: np.ndarray | list[float]) -> np.ndarray:
        n_omega = len(omega_array)
        spectrum = np.empty((self.electron.n_k, n_omega))
        
        count = 0
        progress_bar = ProgressBar('Spectrum', n_omega)
        for omega in omega_array:
            spectrum[..., count] = self.calculate_spectrum(omega)
            
            count += 1
            progress_bar.print(count)

        return spectrum