import numpy as np
from unittest import TestCase

from elphem.common.unit import Length
from elphem.lattice.lattice import Lattice
from elphem.electron.free import FreeElectron
from elphem.phonon.debye import DebyePhonon
from elphem.elph.electron_phonon import ElectronPhonon

class TestUnit(TestCase):
    def setUp(self) -> None:
        a = 2.98 * Length.ANGSTROM["->"]
        self.debye_temperature = 344.0

        temperature = 0.3 * self.debye_temperature

        self.lattice = Lattice('bcc', 'Li', a, temperature)

    def test_with_path(self):
        k_names = ["G", "H"]
        n_split = 20
        k_path = self.lattice.reciprocal.get_path(k_names, n_split)

        n_band = 4
        n_q = np.full(3, 5)

        electron = FreeElectron.create_from_k(self.lattice, 1, n_band, k_path.values)
        phonon = DebyePhonon.create_from_n(self.lattice, self.debye_temperature, n_q)

        electron_phonon = ElectronPhonon(electron, phonon)

        
        omega_array = np.linspace(-1.0, 2.0, 100)
        
        spectrum = electron_phonon.calculate_spectrum_over_range(omega_array)
        
        self.assertEqual(spectrum.shape, (len(k_path.values), len(omega_array)))