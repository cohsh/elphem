from unittest import TestCase
from elphem.common.unit import Length
from elphem.lattice.lattice import Lattice
from elphem.electron.free import FreeElectron

class TestUnit(TestCase):
    def setUp(self) -> None:
        a = 2.98 * Length.ANGSTROM["->"]
        lattice = Lattice('bcc', 'Li', a)

        self.electron = FreeElectron(lattice, n_band=20, n_electron=4, n_k_array=[8,8,8])

    def test_eigenenergies(self):
        eigenenergies = self.electron.get_eigenenergies()
        correct_shape = (len(self.electron.g), len(self.electron.k))
        self.assertEqual(eigenenergies.shape, correct_shape)

    def test_eigenenergies_with_path(self):
        k_names = ["G", "H", "N", "G", "P", "H"]
        eig_path = self.electron.get_eigenenergies_with_path(k_names, n_split=20)
        
        self.assertEqual(eig_path.values.shape, (self.electron.n_band, len(eig_path.minor_scales)))
        self.assertEqual(len(k_names), len(eig_path.major_scales))