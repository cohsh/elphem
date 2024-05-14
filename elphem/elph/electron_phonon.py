import numpy as np

from elphem.common.unit import Energy
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

    def __init__(self, electron: FreeElectron, phonon: DebyePhonon, sigma: float = 0.00001, eta: float = 0.0001):
        self.n_dim = electron.lattice.n_dim
        self.eigenenergies = electron.eigenenergies
        
        g1, g2, k, q = self.create_ggkq_grid(electron, phonon)

        self.electron = electron.clone_with_gk_grid(g1, k)
        self.electron_inter = electron.clone_with_gk_grid(g2, k + q)
        self.phonon = phonon.clone_with_q_grid(q)
        
        self.green_function = GreenFunction(self.electron_inter, self.phonon, sigma, eta)

        self.coupling2 = np.abs(self.calculate_couplings()) ** 2

    def create_ggkq_grid(self, electron: FreeElectron, phonon: DebyePhonon) -> tuple:
        shape = (electron.n_band, electron.n_band, electron.n_k, phonon.n_q, self.n_dim)
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
        couplings = -1.0j * self.effective_potential * np.nansum((self.phonon.q + self.electron.g - self.electron_inter.g) * self.phonon.eigenvectors, axis=-1) * self.phonon.zero_point_lengths

        return couplings

    def calculate_self_energies(self, temperature: float, omega: float) -> np.ndarray:
        """Calculate a single value of Fan self-energy for given wave vectors.

        Args:
            omega (float): a frequency.

        Returns:
            complex: The Fan self-energy term as a complex number.
        """
        
        return np.nansum(self.coupling2 * self.green_function.calculate(temperature, omega), axis=(1, 3)) / self.phonon.n_q

    def calculate_spectrum(self, temperature: float, omega: float) -> np.ndarray:
        self_energies = self.calculate_self_energies(temperature, omega)
        
        numerator = - self_energies.imag / np.pi
        
        denominator = (omega - self.eigenenergies - self_energies.real) ** 2 + self_energies.imag ** 2
        
        return np.nansum(safe_divide(numerator, denominator), axis=0)

    def calculate_self_energies_over_range(self, temperature: float, omega_array: np.ndarray | list[float]) -> np.ndarray:
        n_omega = len(omega_array)
        self_energies = np.empty(self.eigenenergies.shape + (n_omega,), dtype='complex')
        
        count = 0
        progress_bar = ProgressBar('Self Energy', n_omega)
        for omega in omega_array:
            self_energies[..., count] = self.calculate_self_energies(temperature, omega)
            
            count += 1
            progress_bar.print(count)

        return self_energies
        
    def calculate_spectrum_over_range(self, temperature: float, omega_array: np.ndarray | list[float]) -> np.ndarray:
        n_omega = len(omega_array)
        spectrum = np.empty((self.electron.n_k, n_omega))
        
        count = 0
        progress_bar = ProgressBar('Spectrum', n_omega)
        for omega in omega_array:
            spectrum[..., count] = self.calculate_spectrum(temperature, omega)
            
            count += 1
            progress_bar.print(count)

        return spectrum

    def calculate_coupling_strengths(self, temperature: float, delta_omega: float = 0.000001) -> np.ndarray:
        coupling_strengths = np.empty(self.eigenenergies.shape)
        for i in range(self.electron.n_band):
            for j in range(self.electron.n_k):
                self_energies_plus = self.calculate_self_energies(temperature, self.eigenenergies[i,j] + delta_omega)
                self_energies_minus = self.calculate_self_energies(temperature, self.eigenenergies[i,j] - delta_omega)
                coupling_strengths[i,j] = - (self_energies_plus[i,j].real - self_energies_minus[i,j].real) / (2.0 * delta_omega)
        return coupling_strengths

    def calculate_heat_capacity(self, temperature: float, n_omega: int, delta_temperature: float = 0.01) -> float:
        entropy_plus = self.calculate_entropy(temperature + delta_temperature, n_omega)
        entropy_minus = self.calculate_entropy(temperature - delta_temperature, n_omega)

        return (entropy_plus - entropy_minus) / (2.0 * delta_temperature) * temperature

    def calculate_entropy(self, temperature: float, n_omega: int) -> np.ndarray:
        kbt = temperature * Energy.KELVIN['->']
        omega_cut = kbt

        omega_array = np.linspace(0.0, omega_cut, n_omega)
        delta_omega = omega_cut / n_omega

        self_energies = self.calculate_self_energies_over_range(temperature, omega_array)
        
        shape = self_energies.shape + (n_omega,)
        
        omega_array_broadcast = np.broadcast_to(omega_array[np.newaxis, np.newaxis, :], shape)
        self_energies_broadcast = np.broadcast_to(self_energies[:, :, np.newaxis], shape)
        
        cosh2 = np.cosh(omega_array_broadcast / (2 * kbt)) ** 2
        
        coefficient = self.electron.calculate_dos(0.0) * Energy.KELVIN['->'] / kbt ** 2 / self.electron.n_k * delta_omega
        
        entropy = coefficient * np.nansum(omega_array_broadcast * (omega_array_broadcast - self_energies_broadcast.real) / cosh2)
        
        return entropy

    def calculate_heat_capacity_without_couplings_analytical(self, temperature: float) -> float:
        return 2.0 * (np.pi * Energy.KELVIN['->']) ** 2 * self.electron.calculate_dos(0.0) * temperature / 3.0

    def calculate_heat_capacity_without_couplings(self, temperature: float, n_omega: int, delta_temperature: float = 0.01) -> float:
        entropy_plus = self.calculate_entropy_without_couplings(temperature + delta_temperature, n_omega)
        entropy_minus = self.calculate_entropy_without_couplings(temperature - delta_temperature, n_omega)
        
        return (entropy_plus - entropy_minus) / (2.0 * delta_temperature) * temperature

    def calculate_entropy_without_couplings(self, temperature: float, n_omega: int) -> float:
        kbt = temperature * Energy.KELVIN['->']
        omega_cut = kbt * 10.0

        omega_array = np.linspace(0.0, omega_cut, n_omega)
        delta_omega = omega_cut / n_omega
        
        cosh2 = np.cosh(omega_array / (2 * kbt)) ** 2
        
        coefficient = self.electron.calculate_dos(0.0) * Energy.KELVIN['->'] / kbt ** 2 * delta_omega
        
        entropy = coefficient * np.nansum(omega_array ** 2 / cosh2)
        
        return entropy
