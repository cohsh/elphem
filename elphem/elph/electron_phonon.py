import numpy as np

from elphem.electron.electron import Electron
from elphem.phonon.phonon import Phonon
from elphem.elph.green_function import GreenFunction

from elphem.common.stdout import ProgressBar
from elphem.common.function import safe_divide


class ElectronPhonon:
    """Calculate the electron-phonon components for electronic states in a lattice.

    Attributes:
        electron (Electron): Free electron in a given empty lattice.
        phonon (Phonon): Debye Phonon.
        temperature (float): Temperature of the system in Kelvin.
        sigma (float): Smearing parameter for the Gaussian distribution, defaults to 0.01.
        eta (float): Small positive constant to ensure numerical stability, defaults to 0.01.
        effective_potential (float): Effective potential used in electron-phonon coupling calculation, defaults to 1.0 / 16.0.
    """
    def __init__(self, electron: Electron, phonon: Phonon, temperature: float, 
                sigma: float = 0.001, eta: float = 0.0005,
                effective_potential: float = 1.0 / 16.0, n_bands: int = 1):
        self.temperature = temperature
        self.effective_potential = effective_potential
        self.n_bands = n_bands
        self.n_dim = electron.lattice.n_dim
        self.eigenenergies = electron.eigenenergies[0:self.n_bands, :]
        
        g1, g2, k, q = self.create_ggkq_grid(electron, phonon)

        self.electron = electron.clone_with_gk_grid(g1, k)
        self.electron_inter = electron.clone_with_gk_grid(g2, k + q)
        self.phonon = phonon.clone_with_q_grid(q)
        
        self.green_function = GreenFunction(self.electron_inter, self.phonon, self.temperature, sigma, eta)

        self.coupling2 = np.abs(self.calculate_couplings()) ** 2

    def create_ggkq_grid(self, electron: Electron, phonon: Phonon) -> tuple:
        shape = (self.n_bands, electron.n_bands, electron.n_k, phonon.n_q, self.n_dim)
        
        g1 = np.broadcast_to(electron.g[:self.n_bands, np.newaxis, np.newaxis, np.newaxis, :], shape)
        g2 = np.broadcast_to(electron.g[np.newaxis, :, np.newaxis, np.newaxis, :], shape)
        k = np.broadcast_to(electron.k[np.newaxis, np.newaxis, :, np.newaxis, :], shape)
        q = np.broadcast_to(phonon.q[np.newaxis, np.newaxis, np.newaxis, :, :], shape)
        
        return g1, g2, k, q

    def calculate_couplings(self) -> np.ndarray:
        """Calculate the lowest-order electron-phonon coupling between states.

        Returns:
            np.ndarray: The electron-phonon coupling strength for the given vectors.
        """
        return -1.0j * self.effective_potential * np.nansum((self.phonon.q + self.electron.g - self.electron_inter.g) * self.phonon.eigenvectors, axis=-1) * self.phonon.zero_point_lengths

    def calculate_self_energies(self, omega: float) -> np.ndarray:
        """Calculate a single value of Fan self-energy for given wave vectors.

        Args:
            omega (float): a frequency.

        Returns:
            complex: The Fan self-energy term as a complex number.
        """
        
        return np.nansum(self.coupling2 * self.green_function.calculate(omega), axis=(1, 3)) / self.phonon.n_q

    def calculate_electron_phonon_renormalization(self) -> np.ndarray:
        epr = np.empty(self.eigenenergies.shape)
        
        count = 0
        progress_bar = ProgressBar('Electron Phonon Renormalization', self.n_bands * self.electron.n_k)
        for i in range(self.n_bands):
            for j in range(self.electron.n_k):
                epr[i, j] = (self.calculate_self_energies(self.eigenenergies[i, j])).real
                count += 1
                progress_bar.print(count)

        return epr

    def calculate_spectrum(self, omega: float) -> np.ndarray:
        self_energies = self.calculate_self_energies(omega)
        
        numerator = - self_energies.imag / np.pi
        
        denominator = (omega - self.eigenenergies - self_energies.real) ** 2 + self_energies.imag ** 2
        
        return np.nansum(safe_divide(numerator, denominator), axis=0)

    def calculate_self_energies_over_range(self, omega_array: np.ndarray | list[float]) -> np.ndarray:
        n_omega = len(omega_array)
        self_energies = np.empty(self.eigenenergies.shape + (n_omega,), dtype='complex')
        
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

    def calculate_coupling_strengths(self, delta_omega: float = 0.000001) -> np.ndarray:
        coupling_strengths = np.empty(self.eigenenergies.shape)
        
        count = 0
        progress_bar = ProgressBar('Coupling Strength', self.n_bands * self.electron.n_k)
        for i in range(self.n_bands):
            for j in range(self.electron.n_k):
                self_energies_plus = self.calculate_self_energies(self.eigenenergies[i,j] + delta_omega)
                self_energies_minus = self.calculate_self_energies(self.eigenenergies[i,j] - delta_omega)
                coupling_strengths[i,j] = - (self_energies_plus[i,j].real - self_energies_minus[i,j].real) / (2.0 * delta_omega)
                
                count += 1
                progress_bar.print(count)
        
        return coupling_strengths