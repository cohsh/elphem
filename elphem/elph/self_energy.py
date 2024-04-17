import numpy as np
from dataclasses import dataclass

from elphem.lattice.empty import EmptyLattice
from elphem.electron.free import FreeElectron
from elphem.phonon.debye import DebyeModel
from elphem.elph.distribution import fermi_distribution, bose_distribution, gaussian_distribution, safe_divide

@dataclass
class SelfEnergy:
    """Calculate the self-energy components for electronic states in a lattice.

    Attributes:
        lattice (EmptyLattice): The crystal lattice being studied.
        electron (FreeElectron): Free electron model for the lattice.
        phonon (DebyeModel): Phonon model using Debye approximation.
        temperature (float): Temperature of the system in Kelvin.
        sigma (float): Smearing parameter for the Gaussian distribution, defaults to 0.01.
        eta (float): Small positive constant to ensure numerical stability, defaults to 0.01.
        effective_potential (float): Effective potential used in electron-phonon coupling calculation, defaults to 1.0 / 16.0.
    """
    lattice: EmptyLattice
    electron: FreeElectron
    phonon: DebyeModel
    temperature: float
    sigma: float = 0.01
    eta: float = 0.01
    effective_potential: float = 1.0 / 16.0
    
    def coupling(self, g1: np.ndarray, g2: np.ndarray, q: np.ndarray) -> np.ndarray:
        """Calculate the lowest-order electron-phonon coupling between states.

        Args:
            g1 (np.ndarray): Initial G-vector in reciprocal space.
            g2 (np.ndarray): Final G-vector in reciprocal space.
            q (np.ndarray): Phonon wave vector in reciprocal space.

        Returns:
            np.ndarray: The electron-phonon coupling strength for the given vectors.
        """
        q_norm = np.linalg.norm(q, axis=-1)
        delta_g = g1 - g2
        q_dot = np.sum(q * delta_g, axis=-1) 

        mask = q_norm > 0
        result = np.zeros_like(q_norm)
        
        denominator = np.sqrt(2.0 * self.phonon.mass * self.phonon.speed) * q_norm ** 1.5
        result[mask] = safe_divide(self.effective_potential * q_dot[mask], denominator[mask])
        
        return result

    def calculate_fan_term(self, g: np.ndarray, k: np.ndarray, n_q: np.ndarray) -> complex:
        """Calculate a single value of Fan self-energy for given wave vectors.

        Args:
            g (np.ndarray): G-vector in reciprocal space.
            k (np.ndarray): k-vector of the electron state.
            n_q (np.ndarray): Density of intermediate q-vectors for integration.

        Returns:
            complex: The Fan self-energy term as a complex number.
        """        
        g_inter, q = self.electron.grid(n_q) # Generate intermediate G, q grid.

        omega = self.phonon.eigenenergy(q)
        
        coeff = 2.0 * np.pi / np.prod(n_q)

        epsilon = self.electron.eigenenergy(k + g)
        epsilon_inter = self.electron.eigenenergy(k + g_inter + q)

        fermi = fermi_distribution(self.temperature, epsilon_inter)
        bose = bose_distribution(self.temperature, omega)

        coupling = self.coupling(g, g_inter, q)
    
        delta_energy = epsilon - epsilon_inter
        # Real Part
        green_part_real = (safe_divide(1.0 - fermi + bose, delta_energy - omega + self.eta * 1.0j)
                            + safe_divide(fermi + bose, delta_energy + omega + self.eta * 1.0j)).real

        # Imaginary Part
        green_part_imag = ((1.0 - fermi + bose) * gaussian_distribution(self.sigma, delta_energy - omega)
                        + (fermi + bose) * gaussian_distribution(self.sigma, delta_energy + omega))

        selfen = (np.nansum(np.abs(coupling) ** 2 * green_part_real) 
                        + 1.0j * np.nansum(np.abs(coupling) ** 2 * green_part_imag))
        
        return selfen * coeff

    def calculate_coupling_strength(self, g: np.ndarray, k: np.ndarray, n_q: np.ndarray) -> float:
        """Calculate the electron-phonon coupling strength for given wave vectors.

        Args:
            g (np.ndarray): G-vector in reciprocal space.
            k (np.ndarray): k-vector of the electron state.
            n_q (np.ndarray): Density of q-vectors for the integration.

        Returns:
            float: The calculated electron-phonon coupling strength.
        """
        
        g_inter, q = self.electron.grid(n_q) # Generate intermediate G, q grid.

        omega = self.phonon.eigenenergy(q)
        bose = bose_distribution(self.temperature, omega)
        
        coeff = 2.0 * np.pi / np.prod(n_q)

        epsilon = self.electron.eigenenergy(k + g)
        epsilon_inter = self.electron.eigenenergy(k + g_inter + q)

        fermi = fermi_distribution(self.temperature, epsilon_inter)

        coupling = self.coupling(g, g_inter, q)
    
        delta_energy = epsilon - epsilon_inter
        # Real Part
        partial_green_part_real = - (safe_divide(1.0 - fermi + bose, (delta_energy - omega + self.eta * 1.0j) ** 2)
                                    + safe_divide(fermi + bose, (delta_energy + omega + self.eta * 1.0j) ** 2)).real

        coupling_strength = - np.nansum(np.abs(coupling) ** 2 * partial_green_part_real)
        
        return coupling_strength * coeff
    
    def calculate_qp_strength(self, g: np.ndarray, k: np.ndarray, n_q: np.ndarray) -> float:
        """Calculate the quasiparticle strength for given wave vectors.

        Args:
            g (np.ndarray): G-vector in reciprocal space.
            k (np.ndarray): k-vector of the electron state.
            n_q (np.ndarray): Density of q-vectors for the integration.

        Returns:
            float: The quasiparticle strength.
        """
        coupling_strength = self.calculate_coupling_strength(g, k, n_q)
        qp_strength = safe_divide(1.0, 1.0 + coupling_strength)

        return qp_strength