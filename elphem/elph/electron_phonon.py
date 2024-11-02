import numpy as np
import tqdm

from elphem.electron.electron import Electron
from elphem.phonon.phonon import Phonon
from elphem.elph.green_function import GreenFunction


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
                sigma: float = 0.0001, eta: float = 0.0001,
                coupling_type: str = "bardeen", cutoff: float = np.inf):

        self.sigma = sigma
        self.eta = eta
        self.gaussian_coefficient_a = 2.0 * self.sigma ** 2
        self.gaussian_coefficient_b = np.sqrt(2.0 * np.pi) * self.sigma
        
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
        self.coupling2 = np.abs(self.calculate_couplings(coupling_type, self.electron_inter, self.electron, self.phonon, cutoff=cutoff)) ** 2

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

    def calculate_couplings(self, coupling_type: str,
                            electron_out: Electron, electron_in: Electron, phonon: Phonon,
                            cutoff: float = np.inf) -> np.ndarray:
        if coupling_type == "bloch":
            return self.calculate_couplings_bloch(electron_out, electron_in, phonon, cutoff=cutoff)
        elif coupling_type == "nordheim":
            return self.calculate_couplings_nordheim(electron_out, electron_in, phonon, cutoff=cutoff)
        elif coupling_type == "bardeen":
            return self.calculate_couplings_bardeen(electron_out, electron_in, phonon, cutoff=cutoff)
        else:
            raise ValueError("coupling_type is invalid.")

    def calculate_couplings_bloch(self, electron_out: Electron, electron_in: Electron, phonon: Phonon, cutoff: float = np.inf) -> np.ndarray:
        """Calculate the Bloch (1929) electron-phonon coupling constants.

        Returns:
            np.ndarray: The lowest-order electron-phonon coupling constants
        """
        potential = 1.0 / 16.0
        
        couplings = -1.0j * potential * np.nansum((phonon.q + electron_out.g - electron_in.g) * phonon.eigenvectors, axis=-1) * phonon.zero_point_lengths
        
        couplings = np.where(np.abs(couplings) < cutoff, couplings, 0.0+0.0j)

        return couplings

    def calculate_couplings_nordheim(self, electron_out: Electron, electron_in: Electron, phonon: Phonon, cutoff: float = np.inf) -> np.ndarray:
        """Calculate the Nordheim (1931) electron-phonon coupling constants.

        Returns:
            np.ndarray: The lowest-order electron-phonon coupling constants
        """
        wave_vector = electron_out.g - electron_in.g + phonon.q
        potential = 4.0 / self.electron.lattice.primitive.volume * electron_out.n_electrons * np.pi / ( np.nansum(wave_vector, axis=-1) ** 2)
        
        couplings = -1.0j * potential * np.nansum(wave_vector * self.phonon.eigenvectors, axis=-1) * self.phonon.zero_point_lengths
        
        couplings = np.where(np.abs(couplings) < cutoff, couplings, 0.0+0.0j)

        return couplings


    def calculate_couplings_bardeen(self, electron_out: Electron, electron_in: Electron, phonon: Phonon, cutoff: float = np.inf) -> np.ndarray:
        """Calculate the Bardeen (1937) electron-phonon coupling constants.

        Returns:
            np.ndarray: The lowest-order electron-phonon coupling constants
        """
        wave_vector = electron_out.g - electron_in.g + phonon.q
        
        wave_number = np.nansum(wave_vector, axis=-1)

        numerator = 4.0 * electron_out.n_electrons * np.pi
        denominator = wave_number ** 2 + electron_out.thomas_fermi_wave_number ** 2 * self.calculate_lindhard_function(wave_number / (2.0 * electron_out.fermi_wave_number))
        
        couplings = -1.0j / electron_out.lattice.primitive.volume * numerator / denominator * np.nansum(wave_vector * phonon.eigenvectors, axis=-1) * phonon.zero_point_lengths
    
        couplings = np.where(np.abs(couplings) < cutoff, couplings, 0.0+0.0j)

        return couplings
    
    @staticmethod
    def calculate_lindhard_function(x: np.ndarray) -> np.ndarray:
        return 0.5 + (1.0 - x ** 2) / (4.0 * x) * np.log(np.abs(1.0 + x) / np.abs(1.0 - x))

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
        epr = np.empty(self.eigenenergies.shape, dtype='complex')
        
        # calculate EPR
        for i in tqdm.tqdm(range(self.n_bands)):
            for j in tqdm.tqdm(range(self.electron.n_k), leave=False):
                self_energies = self.calculate_self_energies(self.eigenenergies[i, j])
                epr[i, j] = self_energies[i, j]

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
        return np.nansum(numerator / denominator, axis=0)

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
        for i in tqdm.tqdm(range(n_omega)):
            self_energies[..., i] = self.calculate_self_energies(omega_array[i])

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
        for i in tqdm.tqdm(range(n_omega)):
            spectrum[..., i] = self.calculate_spectrum(omega_array[i])
        
        # normalization
        if normalize:
            spectrum_sum = np.nansum(spectrum, axis=1)
            for i in range(n_omega):
                spectrum[..., i] /= spectrum_sum

        return spectrum
    
    def calculate_green_functions(self, omega: float | np.ndarray):
        real_part = self.calculate_green_functions_real(omega)
        imag_part = self.calculate_green_functions_imag(omega)
        
        return real_part + 1.0j * imag_part
    
    def calculate_green_functions_real(self, omega: float | np.ndarray):
        return (1.0 / (omega + 1.0j * self.eta)).real

    def calculate_green_functions_imag(self, omega: float | np.ndarray):
        return - np.pi * np.exp(- omega ** 2 / self.gaussian_coefficient_a) / self.gaussian_coefficient_b