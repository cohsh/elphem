import numpy as np

from elphem.common.stdout import ProgressBar
from elphem.common.function import safe_divide
from elphem.electron.free import FreeElectron
from elphem.phonon.debye import DebyePhonon
from elphem.elph.green_function import GreenFunction

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
    effective_potential: float = 1.0 / 16.0

    def __init__(self, electron: FreeElectron, phonon: DebyePhonon):
        self.coefficient = 1.0 / phonon.n_q
        self.eigenenergies = electron.eigenenergies
        
        g1, g2, k, q = self.create_ggkq_grid(electron, phonon)

        self.electron = electron.clone_with_gk_grid(g1, k)
        self.electron_inter = electron.clone_with_gk_grid(g2, k + q)
        self.phonon = phonon.clone_with_q(q)
        
        self.green_function = GreenFunction(self.electron_inter, self.phonon)

        self.coupling2 = np.abs(self.calculate_couplings()) ** 2

    def create_ggkq_grid(self, electron: FreeElectron, phonon: DebyePhonon) -> tuple:
        shape = (electron.n_band, electron.n_band, electron.n_k, phonon.n_q, 3)
        
        g1 = np.broadcast_to(electron.g[:, np.newaxis, np.newaxis, np.newaxis, :], shape)
        g2 = np.broadcast_to(electron.g[np.newaxis, :, np.newaxis, np.newaxis, :], shape)
        k = np.broadcast_to(electron.k[np.newaxis, np.newaxis, :, np.newaxis, :], shape)
        q = np.broadcast_to(phonon.q[np.newaxis, np.newaxis, np.newaxis, :, :], shape)
        
        return g1, g2, k, q

    def calculate_couplings(self) -> np.ndarray:
        """Calculate the lowest-order electron-phonon coupling between states.

        Returns:
            np.ndarray: The electron-phonon coupling strength for the given vectors.
        """
        
        couplings = -1.0j * self.effective_potential * np.sum((self.phonon.q + self.electron.g - self.electron_inter.g) * self.phonon.eigenvectors, axis=-1) * self.phonon.zero_point_lengths

        return couplings

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
        
        denominator = (omega - self.eigenenergies - self_energies.real) ** 2 + self_energies.imag ** 2
        
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