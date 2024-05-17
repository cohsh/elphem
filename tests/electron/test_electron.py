from unittest import TestCase
import numpy as np
from elphem.common.unit import Length
from elphem.lattice.lattice import Lattice3D
from elphem.electron.electron import Electron

class TestUnit(TestCase):
    def setUp(self) -> None:
        a = 2.98 * Length.ANGSTROM["->"]
        self.lattice = Lattice3D('bcc', 'Li', a)

    def test_create_from_n(self):
        electron = Electron.create_from_n(self.lattice, n_electrons=1, n_bands=20, n_k_array=[8,8,8])
        eigenenergies = electron.eigenenergies
        
        correct_shape = (len(electron.g), len(electron.k))
        self.assertEqual(eigenenergies.shape, correct_shape)

    def test_create_from_k(self):
        k = np.array([0., 0., 0.])
        electron = Electron.create_from_k(self.lattice, n_electrons=1, n_bands=20, k_array=k)
        eigenenergies = electron.eigenenergies
        
        correct_shape = (len(electron.g), len(electron.k))
        self.assertEqual(eigenenergies.shape, correct_shape)

    def test_create_from_path(self):
        k_names = ["G", "H", "N", "G", "P", "H"]
        k_path = self.lattice.get_k_path(k_names, 20)
        
        electron = Electron.create_from_path(self.lattice, n_electrons=1, n_bands=20, k_path=k_path)
        eigenenergies = electron.eigenenergies
        
        correct_shape = (len(electron.g), len(k_path.values))
        self.assertEqual(eigenenergies.shape, correct_shape)