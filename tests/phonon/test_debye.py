from unittest import TestCase
import numpy as np
from elphem.const.unit import Length
from elphem.lattice.lattice import Lattice
from elphem.phonon.debye import DebyePhonon

class TestUnit(TestCase):
    def setUp(self) -> None:
        a = 2.98 * Length.ANGSTROM["->"]
        debye_temperature = 344.0

        lattice = Lattice('bcc', 'Li', a)

        self.phonon = DebyePhonon(lattice, debye_temperature)
    
    def test_dispersion(self):
        q_names = ["G", "H", "N", "G", "P", "H"]
        x, omega, x_special = self.phonon.get_dispersion(q_names, n_split=20)

        self.assertEqual(len(omega), len(x))
        self.assertEqual(len(q_names), len(x_special))