import numpy as np

from elphem.electron.electron import Electron
from elphem.phonon.phonon import Phonon
from elphem.elph.green_function import GreenFunction

from elphem.common.stdout import ProgressBar
from elphem.common.function import safe_divide


class ElectronPhonon:
    """Calculate the electron-phonon interactions.

    Attributes:
        temperature (float): Temperature of the system in Kelvin
        sigma (float): Smearing parameter for the Gaussian distribution, defaults to 0.001 Hartree
        eta (float): Small positive constant to ensure numerical stability, defaults to 0.0005 Hartree
        effective_potential (float): Effective potential used in electron-phonon coupling calculation, defaults to 0.0625
        n_bands (int): Number of bands for calculating self energies
        electron (Electron): Free electron for initial and final states
        electron_inter (Electron): Free electron for intermediate states
        phonon (Phonon): Debye Phonon
        green_function (GreenFunction): Green function
        couplings (np.ndarray): electron-phonon coupling constants squared
    """
    def __init__(self, electron: Electron, phonon: Phonon, temperature: float, n_bands: int,
                sigma: float = 0.001, eta: float = 0.0005, coupling_type: str = "bloch"):
        self.n_dim = electron.lattice.n_dim
        self.temperature = temperature
        if n_bands > electron.n_bands:
            self.n_bands = electron.n_bands
        else:
            self.n_bands = n_bands

        self.eigenenergies = electron.eigenenergies[0:self.n_bands, :]
        
        g1, g2, k, q = self.create_ggkq_grid(electron, phonon)

        # set electrons and a phonon
        self.electron = electron.clone_with_gk_grid(g1, k)
        self.electron_inter = electron.clone_with_gk_grid(g2, k + q)
        self.phonon = phonon.clone_with_q_grid(q)
        
        # set Green function
        self.green_function = GreenFunction(self.electron_inter, self.phonon, self.temperature, sigma, eta)

        # set electron-phonon coupling constants squared
        self.coupling2 = np.abs(self.calculate_couplings(coupling_type)) ** 2

    def create_ggkq_grid(self, electron: Electron, phonon: Phonon) -> tuple:
        """Create (G_!, G_2, k, q) combined grids

        Args:
            electron (Electron): Free electron
            phonon (Phonon): Debye phonon

        Returns:
            tuple: (G_1, G_2, k, q) combined grids
        """
        shape = (self.n_bands, electron.n_bands, electron.n_k, phonon.n_q, self.n_dim)
        
        g1 = np.broadcast_to(electron.g[:self.n_bands, np.newaxis, np.newaxis, np.newaxis, :], shape)
        g2 = np.broadcast_to(electron.g[np.newaxis, :, np.newaxis, np.newaxis, :], shape)
        k = np.broadcast_to(electron.k[np.newaxis, np.newaxis, :, np.newaxis, :], shape)
        q = np.broadcast_to(phonon.q[np.newaxis, np.newaxis, np.newaxis, :, :], shape)
        
        return g1, g2, k, q

    def calculate_couplings(self, coupling_type: str) -> np.ndarray:
        if coupling_type == "bloch":
            return self.calculate_couplings_bloch()
        elif coupling_type == "nordheim":
            return self.calculate_couplings_nordheim()
        else:
            raise ValueError("coupling_type is invalid.")

    def calculate_couplings_bloch(self) -> np.ndarray:
        """Calculate the Bloch (1929) electron-phonon coupling constants.

        Returns:
            np.ndarray: The lowest-order electron-phonon coupling constants
        """
        potential = 1.0 / 16.0
        
        return -1.0j * potential * np.nansum((self.phonon.q + self.electron.g - self.electron_inter.g) * self.phonon.eigenvectors, axis=-1) * self.phonon.zero_point_lengths

    def calculate_couplings_nordheim(self) -> np.ndarray:
        """Calculate the Nordheim (1931) electron-phonon coupling constants.

        Returns:
            np.ndarray: The lowest-order electron-phonon coupling constants
        """
        potential = 4.0 * self.electron.n_electrons * np.pi / ( np.nansum(self.phonon.q + self.electron.g - self.electron_inter.g, axis=-1) ** 2)
        
        return -1.0j * potential * np.nansum((self.phonon.q + self.electron.g - self.electron_inter.g) * self.phonon.eigenvectors, axis=-1) * self.phonon.zero_point_lengths

    def calculate_self_energies(self, omega: float) -> np.ndarray:
        """Calculate Fan self-energies.

        Args:
            omega (float): a frequency.

        Returns:
            np.ndarray: A numpy array of Fan self-energies.
        """
        
        return np.nansum(self.coupling2 * self.green_function.calculate(omega), axis=(1, 3)) / self.phonon.n_q

    def calculate_electron_phonon_renormalization(self) -> np.ndarray:
        """Calculate electron-phonon renormalizations of electron eigenenergies (EPR).

        Returns:
            np.ndarray: A numpy array of EPR
        """
        # prepare an array for EPR
        epr = np.empty(self.eigenenergies.shape)
        
        # calculate EPR
        count = 0
        progress_bar = ProgressBar('Electron Phonon Renormalization', self.n_bands * self.electron.n_k)
        for i in range(self.n_bands):
            for j in range(self.electron.n_k):
                self_energies = self.calculate_self_energies(self.eigenenergies[i, j])
                epr[i, j] = self_energies[i, j].real
                count += 1
                progress_bar.print(count)

        return epr

    def calculate_spectrum(self, omega: float) -> np.ndarray:
        """Calculate the spectral function for a frequency.

        Args:
            omega (float): a frequency

        Returns:
            np.ndarray: A numpy array of the spectral function at omega Hartree
        """
        # calculate self energies at omega
        self_energies = self.calculate_self_energies(omega)
        
        # calculate numerator and denominator separately
        numerator = - self_energies.imag / np.pi
        denominator = (omega - self.eigenenergies - self_energies.real) ** 2 + self_energies.imag ** 2
        
        # sum over bands
        return np.nansum(safe_divide(numerator, denominator), axis=0)

    def calculate_self_energies_over_range(self, omega_array: np.ndarray | list[float]) -> np.ndarray:
        """Calculate self energies over a given array of frequencies.

        Args:
            omega_array (np.ndarray | list[float]): a numpy array or list of frequencies

        Returns:
            np.ndarray: a numpy array of self energies
        """
        # prepare the length of frequency-array and an array for self energies
        n_omega = len(omega_array)
        self_energies = np.empty(self.eigenenergies.shape + (n_omega,), dtype='complex')
        
        # calculate self energies
        count = 0
        progress_bar = ProgressBar('Self Energy', n_omega)
        for omega in omega_array:
            self_energies[..., count] = self.calculate_self_energies(omega)
            
            count += 1
            progress_bar.print(count)

        return self_energies
        
    def calculate_spectrum_over_range(self, omega_array: np.ndarray | list[float], normalize: bool = False) -> np.ndarray:
        """Calculate spectral function over a given array of frequencies.

        Args:
            omega_array (np.ndarray | list[float]): A numpy array or list of frequencies
            normalize (bool, optional): Flag for normalizing spectral functions. Defaults to False.

        Returns:
            np.ndarray: A numpy array of spectral functions
        """
        # prepare
        n_omega = len(omega_array)
        spectrum = np.empty((self.electron.n_k, n_omega))
        
        # calculate spectral functions
        progress_bar = ProgressBar('Spectrum', n_omega)
        for i in range(n_omega):
            spectrum[..., i] = self.calculate_spectrum(omega_array[i])

            progress_bar.print(i)
        
        # normalization
        if normalize:
            spectrum_sum = np.nansum(spectrum, axis=1)
            for i in range(n_omega):
                spectrum[..., i] /= spectrum_sum

        return spectrum

    def calculate_coupling_strengths(self, delta_omega: float = 0.000001) -> np.ndarray:
        """Calculate electron-phonon coupling strengths

        Args:
            delta_omega (float, optional): A small value used for numerical differentiation. Defaults to 0.000001 Hartree.

        Returns:
            np.ndarray: A numpy array of coupling strengths
        """
        # prepare an array
        coupling_strengths = np.empty(self.eigenenergies.shape)
        
        # calculate coupling strengths
        count = 0
        progress_bar = ProgressBar('Coupling Strength', self.n_bands * self.electron.n_k)
        for i in range(self.n_bands):
            for j in range(self.electron.n_k):
                # numerical differentiation
                self_energies_plus = self.calculate_self_energies(self.eigenenergies[i,j] + delta_omega)
                self_energies_minus = self.calculate_self_energies(self.eigenenergies[i,j] - delta_omega)
                coupling_strengths[i,j] = - (self_energies_plus[i,j].real - self_energies_minus[i,j].real) / (2.0 * delta_omega)
                
                count += 1
                progress_bar.print(count)
        
        return coupling_strengths