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
    sigma: float = 0.001
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
        
        eigenenergy = self.phonon.get_eigenenergy(q)
        eigenvector = self.phonon.get_eigenvector(q)

        delta_g = g1 - g2

        numerator = -1.0j * self.effective_potential * np.sum((q + delta_g) * eigenvector, axis=-1)
        denominator = np.sqrt(2.0 * self.phonon.lattice.mass * eigenenergy)

        coupling = safe_divide(numerator, denominator)
        
        return coupling

    def get_self_energy_and_coupling_strength(self, g: np.ndarray, k: np.ndarray, n_q: np.ndarray) -> tuple:
        """Calculate a single value of Fan self-energy for given wave vectors.

        Args:
            g (np.ndarray): G-vector in reciprocal space.
            k (np.ndarray): k-vector of the electron state.
            n_q (np.ndarray): Density of intermediate q-vectors for integration.

        Returns:
            complex: The Fan self-energy term as a complex number.
        """        
        g_inter, q = self.electron.get_gk_grid(n_q) # Generate intermediate G, q grid.

        phonon_eigenenergy = self.phonon.get_eigenenergy(q)
        
        coefficient = 2.0 * np.pi / np.prod(n_q)

        electron_eigenenergy = self.electron.get_eigenenergy(k + g)
        electron_eigenenergy_inter = self.electron.get_eigenenergy(k + g_inter + q)

        fermi = fermi_distribution(self.temperature, electron_eigenenergy_inter)
        bose = bose_distribution(self.temperature, phonon_eigenenergy)

        coupling = self.get_coupling(g, g_inter, q)
    
        delta_energy = electron_eigenenergy - electron_eigenenergy_inter
    
        # Real Part
        green_function_real = (safe_divide(1.0 - fermi + bose, delta_energy - phonon_eigenenergy + self.eta * 1.0j)
                            + safe_divide(fermi + bose, delta_energy + phonon_eigenenergy + self.eta * 1.0j)).real

        # Imaginary Part
        green_function_imag = ((1.0 - fermi + bose) * gaussian_distribution(self.sigma, delta_energy - phonon_eigenenergy)
                        + (fermi + bose) * gaussian_distribution(self.sigma, delta_energy + phonon_eigenenergy))

        # Real Part
        partial_green_function_real = - (safe_divide(1.0 - fermi + bose, (delta_energy - phonon_eigenenergy + self.eta * 1.0j) ** 2)
                                    + safe_divide(fermi + bose, (delta_energy + phonon_eigenenergy + self.eta * 1.0j) ** 2)).real

        self_energy = (np.nansum(np.abs(coupling) ** 2 * green_function_real)
                        + 1.0j * np.nansum(np.abs(coupling) ** 2 * green_function_imag)) * coefficient

        coupling_strength = - np.nansum(np.abs(coupling) ** 2 * partial_green_function_real) * coefficient
        
        return self_energy, coupling_strength

    @staticmethod
    def get_qp_strength(coupling_strength: np.ndarray) -> float:
        """Calculate the quasiparticle strength for given wave vectors.

        Args:
            coupling_strength (np.ndarray): Coupling strength.

        Returns:
            float: The quasiparticle strength.
        """
        qp_strength = safe_divide(1.0, 1.0 + coupling_strength)

        return qp_strength