import numpy as np
from dataclasses import dataclass

from elphem.common.unit import Byte
from elphem.common.function import safe_divide
from elphem.electron.free import FreeElectron
from elphem.phonon.debye import DebyePhonon
from elphem.elph.distribution import fermi_distribution, bose_distribution, gaussian_distribution

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
        
        g1 = g2 = np.broadcast_to(self.electron.reciprocal_vectors[:, np.newaxis, np.newaxis, np.newaxis, :], shape)
        k = np.broadcast_to(k_array[np.newaxis, np.newaxis, :, np.newaxis, :], shape)
        q = np.broadcast_to(q_array[np.newaxis, np.newaxis, np.newaxis, :, :], shape)
        
        print(Byte.get_str(g1.nbytes))
        
        return g1, g2, k, q

    def get_self_energy(self, omega: float, k_array: np.ndarray) -> np.ndarray:
        """Calculate a single value of Fan self-energy for given wave vectors.

        Args:
            omega (float): single value of frequencies.
            k (np.ndarray): k-vectors.

        Returns:
            complex: The Fan self-energy term as a complex number.
        """
        coefficient = 1.0 / np.prod(self.n_qs)
        
        g1, g2, k, q = self.get_ggkq_grid(k_array)
        
        electron_eigenenergy_inter = self.electron.get_eigenenergy(k + q + g2)
        fermi = fermi_distribution(self.temperature, electron_eigenenergy_inter)

        phonon_eigenenergy = self.phonon.get_eigenenergy(q)
        bose = bose_distribution(self.temperature, phonon_eigenenergy)

        coupling = self.get_coupling(g1, g2, q)

        occupation_absorb = 1.0 - fermi + bose
        occupation_emit = fermi + bose
        
        denominator_absorb = omega - electron_eigenenergy_inter - phonon_eigenenergy
        denominator_emit = omega - electron_eigenenergy_inter + phonon_eigenenergy

        green_function_real = (occupation_absorb * self.get_green_function_real(denominator_absorb)
                                + occupation_emit * self.get_green_function_real(denominator_emit))

        green_function_imag = np.pi * (occupation_absorb * self.get_green_function_imag(denominator_absorb)
                                + occupation_emit * self.get_green_function_imag(denominator_emit))

        self_energy = np.nansum(np.abs(coupling) ** 2 * (green_function_real + 1.0j * green_function_imag), axis=(1, 3)) * coefficient
        
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
        
        x, k, special_x = self.electron.lattice.reciprocal_cell.get_path(k_names, n_split)
        eig_array = np.array([self.electron.get_eigenenergy(k + g_i) for g_i in g])

        omega_array = np.linspace(range_omega[0], range_omega[1], n_omega)
        
        for omega in omega_array:
            self_energy = self.get_self_energy(omega, k)
        
        shape = (len(omega_array), self.electron.n_band, len(k))
        
        omega = np.broadcast_to(omega_array[:, np.newaxis, np.newaxis], shape)
        eig = np.broadcast_to(eig_array[np.newaxis, :, :], shape)

        numerator = - self_energy.imag / np.pi
        denominator = (omega - eig - self_energy.real) ** 2 + self_energy.imag ** 2

        fraction = safe_divide(numerator, denominator)

        spectrum = np.nansum(fraction, axis=1)

        return x, omega_array, spectrum, special_x

    def get_green_function_real(self, omega: np.ndarray) -> np.ndarray:
        green_function_real = safe_divide(1.0, omega + self.eta * 1.0j).real
        
        return green_function_real
    
    def get_green_function_imag(self, omega: np.ndarray) -> np.ndarray:
        green_function_imag = gaussian_distribution(self.sigma, omega)
        
        return green_function_imag