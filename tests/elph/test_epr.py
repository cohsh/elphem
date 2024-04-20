import numpy as np
from unittest import TestCase

from elphem.common.unit import Length
from elphem.lattice.lattice import Lattice
from elphem.electron.free import FreeElectron
from elphem.phonon.debye import DebyePhonon
from elphem.elph.electron_phonon import ElectronPhonon
from elphem.elph.epr import EPR

class TestUnit(TestCase):
    def setUp(self) -> None:
        a = 2.98 * Length.ANGSTROM["->"]
        debye_temperature = 344.0
        n_band = 8

        temperature = 0.3 * debye_temperature

        lattice = Lattice('bcc', 'Li', a)
        electron = FreeElectron(lattice, n_band, 1)
        phonon = DebyePhonon(lattice, temperature)

        electron_phonon = ElectronPhonon(electron, phonon, temperature)
        self.epr = EPR(electron_phonon)

    def test_calculate_with_grid(self):
        n_k = np.full(3, 5)        
        n_q = np.full(3, 5)
        
        eig, delta_eig = self.epr.get_with_grid(n_k, n_q)

        self.assertEqual(eig.shape, delta_eig.shape)
    
    def test_calculate_with_path(self):
        k_names = ["G", "H", "N", "G", "P", "H"]
        n_split = 20
        
        n_q = np.array([5]*3)
        
        k, eig, delta_eig, special_k = self.epr.get_with_path(k_names, n_split, n_q)
        
        self.assertEqual(eig.shape, delta_eig.shape)
        self.assertEqual(len(k_names), len(special_k))