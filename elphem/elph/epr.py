import numpy as np
from dataclasses import dataclass

from elphem.elph.electron_phonon import ElectronPhonon

@dataclass
class EPR:
    """A class to calculate the 2nd-order Fan self-energies using the self-energy module.

    Attributes:
        self_energy (SelfEnergy): An instance of SelfEnergy to use for calculations.
    """
    electron_phonon: ElectronPhonon
    
    def get_with_path(self, k_names: list[str], n_split: int) -> tuple:
        """
        Calculate 2nd-order Fan self-energies along a specified path in the Brillouin zone.

        Args:
            k_names (list[str]): A list of special points names defining the path through the Brillouin zone.
            n_split (int): Number of points between each special point to compute the dispersion.
            n_q (np.ndarray): A numpy array specifying the density of q-grid points in each direction.

        Returns:
            tuple: A tuple containing x-coordinates for plotting, eigenenergies, Fan self-energies, and x-coordinates of special points.
        """
        
        g = self.electron_phonon.electron.reciprocal_vectors
        
        x, k, special_x = self.electron_phonon.electron.lattice.reciprocal_cell.get_path(k_names, n_split)
        eig = np.array([self.electron_phonon.electron.get_eigenenergy(k + g_i) for g_i in g])

        self_energy = self.electron_phonon.get_self_energy_rs(k)

        epr = self_energy.real
        
        return x, eig, epr, special_x