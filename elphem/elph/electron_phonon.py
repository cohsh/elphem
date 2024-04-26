import numpy as np
from dataclasses import dataclass

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

    def get_self_energy_rs(self, k_array: np.ndarray) -> np.ndarray:
        """Calculate a single value of Fan self-energy for given wave vectors.

        Args:
            omega (float): single value of frequencies.
            k (np.ndarray): k-vectors.

        Returns:
            complex: The Fan self-energy term as a complex number.
        """
        coefficient = 1.0 / np.prod(self.n_qs)
        
        g1, g2, k, q = self.get_ggkq_grid(k_array)

        electron_eigenenergy = self.electron.get_eigenenergy(k + g1)        
        electron_eigenenergy_inter = self.electron.get_eigenenergy(k + q + g2)
        fermi = fermi_distribution(self.temperature, electron_eigenenergy_inter)

        phonon_eigenenergy = self.phonon.get_eigenenergy(q)
        bose = bose_distribution(self.temperature, phonon_eigenenergy)

        coupling = self.get_coupling(g1, g2, q)

        occupation_absorb = 1.0 - fermi + bose
        occupation_emit = fermi + bose
        
        denominator_absorb = electron_eigenenergy - electron_eigenenergy_inter - phonon_eigenenergy
        denominator_emit = electron_eigenenergy_inter - electron_eigenenergy_inter + phonon_eigenenergy

        green_function_real = (occupation_absorb * self.get_green_function_real(denominator_absorb)
                                + occupation_emit * self.get_green_function_real(denominator_emit))

        green_function_imag = np.pi * (occupation_absorb * self.get_green_function_imag(denominator_absorb)
                                + occupation_emit * self.get_green_function_imag(denominator_emit))

        self_energy = np.nansum(np.abs(coupling) ** 2 * (green_function_real + 1.0j * green_function_imag), axis=(1, 3)) * coefficient
        
        return self_energy

    def get_green_function_real(self, omega: np.ndarray) -> np.ndarray:
        green_function_real = safe_divide(1.0, omega + self.eta * 1.0j).real
        
        return green_function_real
    
    def get_green_function_imag(self, omega: np.ndarray) -> np.ndarray:
        green_function_imag = gaussian_distribution(self.sigma, omega)
        
        return green_function_imag