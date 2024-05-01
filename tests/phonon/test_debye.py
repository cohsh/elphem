from unittest import TestCase
import numpy as np
from elphem.common.unit import Length
from elphem.lattice.lattice import Lattice
from elphem.phonon.debye import DebyePhonon

class TestUnit(TestCase):
    def setUp(self) -> None:
        a = 2.98 * Length.ANGSTROM["->"]
        debye_temperature = 344.0
        n_q_array = [8,8,8]

        self.lattice = Lattice('bcc', 'Li', a)

        self.phonon = DebyePhonon.create_from_n(self.lattice, debye_temperature, n_q_array)
    
    def test_eigenenergies(self):
        omega = self.phonon.eigenenergies
        self.assertEqual(len(omega), self.phonon.n_q)
    
    def test_eigenvectors(self):
        e = self.phonon.eigenvectors
        self.assertEqual(len(e), self.phonon.n_q)
    
    def test_eigenenergies_with_path(self):
        q_names = ["G", "H", "N", "G", "P", "H"]
        q_path = self.lattice.reciprocal.get_path(q_names, 20)
        omega_path = self.phonon.get_eigenenergies_with_path(q_path)

        self.assertEqual(len(omega_path.values), len(omega_path.minor_scales))
        self.assertEqual(len(q_names), len(omega_path.major_scales))