from unittest import TestCase
import numpy as np
from elphem.common.unit import Length
from elphem.lattice.lattice import Lattice3D
from elphem.phonon.phonon import Phonon

class TestUnit(TestCase):
    def setUp(self) -> None:
        a = 2.98 * Length.ANGSTROM["->"]
        self.debye_temperature = 344.0

        self.lattice = Lattice3D('bcc', 'Li', a)

    def test_create_from_q(self):
        q = np.zeros(3)
        phonon = Phonon.create_from_q(self.lattice, self.debye_temperature, q)

        omega = phonon.eigenenergies
        e = phonon.eigenvectors
        self.assertEqual(len(omega), phonon.n_q)
        self.assertEqual(len(e), phonon.n_q)

    def test_create_from_n(self):
        n_q_array = [8,8,8]
        phonon = Phonon.create_from_n(self.lattice, self.debye_temperature, n_q_array)

        omega = phonon.eigenenergies
        e = phonon.eigenvectors
        
        self.assertEqual(len(omega), phonon.n_q)
        self.assertEqual(len(e), phonon.n_q)

    def test_create_from_path(self):
        q_names = ["G", "H", "N", "G", "P", "H"]
        q_path = self.lattice.get_k_path(q_names, 20)
        
        phonon = Phonon.create_from_path(self.lattice, self.debye_temperature, q_path)
        
        omega = phonon.eigenenergies
        e = phonon.eigenvectors

        self.assertEqual(len(omega), phonon.n_q)
        self.assertEqual(len(e), phonon.n_q)