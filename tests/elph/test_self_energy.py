import numpy as np
from unittest import TestCase

from elphem.common.unit import Length
from elphem.lattice.lattice import Lattice3D
from elphem.electron.electron import Electron
from elphem.phonon.phonon import Phonon
from elphem.elph.electron_phonon import ElectronPhonon

class TestUnit(TestCase):
    def setUp(self) -> None:
        n_electrons = 1
        n_bands_electrons = 10
        n_bands_elph = 2
        a = 2.98 * Length.ANGSTROM["->"]
        n_k = np.full(3,4)
        n_q = np.full(3,4)
        
        lattice = Lattice3D('bcc', 'Li', a)

        debye_temperature = 344.0
        temperature = 0.3 * debye_temperature

        self.electron = Electron.create_from_n(lattice, n_electrons, n_bands_electrons, n_k)
        self.phonon = Phonon.create_from_n(lattice, debye_temperature, n_q)
        self.electron_phonon = ElectronPhonon(self.electron, self.phonon, temperature, n_bands_elph)
    
    def test_self_energies(self):
        self_energies = self.electron_phonon.calculate_self_energies(0.0)
        
        self.assertEqual(self_energies.shape, (self.electron_phonon.n_bands, self.electron.n_k))