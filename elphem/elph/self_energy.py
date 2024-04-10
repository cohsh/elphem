import numpy as np
from dataclasses import dataclass

from elphem.lattice.empty import EmptyLattice
from elphem.electron.free import FreeElectron
from elphem.phonon.debye import DebyeModel
from elphem.elph.distribution import fermi_distribution, bose_distribution, gaussian_distribution, safe_divide

@dataclass
class SelfEnergy:
    lattice: EmptyLattice
    electron: FreeElectron
    phonon: DebyeModel
    temperature: float
    sigma: float = 0.01
    eta: float = 0.01
    effective_potential: float = 1.0 / 16.0
    
    def coupling(self, g1: np.ndarray, g2: np.ndarray, q: np.ndarray) -> np.ndarray:
        """
        Calculate lowest-order electron-phonon couplings.
        
        Args
            g1, g2: A numpy array representing G-vector
            k: A numpy array representing k-vector
            q: A numpy array representing k-vector
        
        Return
            A value of the elctron-phonon coupling.
        """
        q_norm = np.linalg.norm(q, axis=-1)
        delta_g = g1 - g2
        q_dot = np.sum(q * delta_g, axis=-1) 

        mask = q_norm > 0
        result = np.zeros_like(q_norm)
        
        denominator = np.sqrt(2.0 * self.phonon.mass * self.phonon.speed) * q_norm ** 1.5
        result[mask] = safe_divide(self.effective_potential * q_dot[mask], denominator[mask])
        
        return result

    def calculate_fan_term(self, n_g: np.ndarray, n_k: np.ndarray, 
                        n_g_inter: np.ndarray, n_q: np.ndarray) -> complex:
        """
        Calculate single values of Fan self-energy.
        
        Args
            g: A numpy array representing G-vector
            k: A numpy array representing k-vector
            n_g: A numpy array representing the dense of intermediate G-vectors
            n_q: A numpy array representing the dense of intermediate q-vectors
            eta: A value of the convergence factor. The default value is 0.01 Hartree.
            
        Return
            A complex value representing Fan self-energy.
        """
        
        g_mesh, k_mesh = self.electron.grid(n_g, n_k)
        
        shape_return = g_mesh[..., 0].shape
        
        g = g_mesh.reshape(-1, 3)
        k = k_mesh.reshape(-1, 3)
        
        g_inter, q = self.electron.grid(n_g_inter, n_q) # Generate intermediate G, q grid.
        omega = self.phonon.eigenenergy(q)
        bose = bose_distribution(self.temperature, omega)
        
        selfen = np.zeros(g[..., 0].shape, dtype=np.complex128)
        coeff = 2.0 * np.pi / np.prod(n_q)

        count = 0
        for g_i, k_i in zip(g, k):
            electron_energy_nk = self.electron.eigenenergy(k_i + g_i)
            electron_energy_mkq = self.electron.eigenenergy(k_i + q + g_inter)

            fermi = fermi_distribution(self.temperature, electron_energy_mkq)

            coupling = self.coupling(g_i, g_inter, q)
        
            delta_energy = electron_energy_nk - electron_energy_mkq
            # Real Part
            green_part_real = ((1.0 - fermi + bose) / (delta_energy - omega + self.eta * 1.0j)
                            + (fermi + bose) / (delta_energy + omega + self.eta * 1.0j)).real

            # Imaginary Part
            green_part_imag = ((1.0 - fermi + bose) * gaussian_distribution(self.sigma, delta_energy - omega)
                            + (fermi + bose) * gaussian_distribution(self.sigma, delta_energy + omega))

            selfen[count] = (np.nansum(np.abs(coupling) ** 2 * green_part_real) 
                             + 1.0j * np.nansum(np.abs(coupling) ** 2 * green_part_imag))
            count += 1
        
        return selfen.reshape(shape_return) * coeff

    def calculate_coupling_strength(self, n_g: np.ndarray, n_k: np.ndarray,
                            n_g_inter: np.ndarray, n_q: np.ndarray) -> float:
        """
        Calculate electron-phonon coupling strengths.
        
        Args
        """
        
        g_mesh, k_mesh = self.electron.grid(n_g, n_k)
        
        shape_return = g_mesh[..., 0].shape
        
        g = g_mesh.reshape(-1, 3)
        k = k_mesh.reshape(-1, 3)
        
        g_inter, q = self.electron.grid(n_g_inter, n_q) # Generate intermediate G, q grid.
        omega = self.phonon.eigenenergy(q)
        bose = bose_distribution(self.temperature, omega)
        
        coupling_strength = np.zeros(g[..., 0].shape, dtype=np.complex128)
        coeff = 2.0 * np.pi / np.prod(n_q)

        count = 0
        for g_i, k_i in zip(g, k):
            electron_energy_nk = self.electron.eigenenergy(k_i + g_i)
            electron_energy_mkq = self.electron.eigenenergy(k_i + q + g_inter)

            fermi = fermi_distribution(self.temperature, electron_energy_mkq)

            coupling = self.coupling(g_i, g_inter, q)
        
            delta_energy = electron_energy_nk - electron_energy_mkq
            # Real Part
            partial_green_part_real = - ((1.0 - fermi + bose) / (delta_energy - omega + self.eta * 1.0j) ** 2
                            + (fermi + bose) / (delta_energy + omega + self.eta * 1.0j) ** 2).real

            coupling_strength[count] = - np.nansum(np.abs(coupling) ** 2 * partial_green_part_real)
            count += 1
        
        return coupling_strength.reshape(shape_return) * coeff
    
    def calculate_qp_strength(self, n_g: np.ndarray, n_k: np.ndarray,
                            n_g_inter: np.ndarray, n_q: np.ndarray) -> float:
        """
        Calculate quasiparticle strengths (z).
        
        Args
        """
        coupling_strength = self.calculate_coupling_strength(n_g, n_k, n_g_inter, n_q)
        
        z = 1.0 / (1.0 + coupling_strength)
        
        return z
