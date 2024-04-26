import numpy as np
from unittest import TestCase

from elphem.common.unit import Length
from elphem.lattice.lattice import Lattice
from elphem.electron.free import FreeElectron
from elphem.phonon.debye import DebyePhonon
from elphem.elph.electron_phonon import ElectronPhonon
from elphem.elph.spectrum import Spectrum

class TestUnit(TestCase):
    def setUp(self) -> None:
        a = 2.98 * Length.ANGSTROM["->"]
        debye_temperature = 344.0
        n_band = 2
        n_q = np.full(3, 4)

        self.range_omega = [-1.0, 2.0]

        temperature = 0.3 * debye_temperature

        lattice = Lattice('bcc', 'Li', a)
        electron = FreeElectron(lattice, n_band, 1)
        phonon = DebyePhonon(lattice, temperature)

        electron_phonon = ElectronPhonon(electron, phonon, temperature, n_q)
        self.spectrum = Spectrum(electron_phonon)

    def test_with_path(self):
        k_names = ["G", "H"]
        n_split = 20
        
        n_omega = 20
        
        k, omegas, a, special_k = self.spectrum.get_with_path(k_names, n_split, n_omega, self.range_omega)
        
        self.assertEqual(a.shape, (len(k), len(omegas)))
        self.assertEqual(len(k_names), len(special_k))