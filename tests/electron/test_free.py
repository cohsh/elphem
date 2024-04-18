from unittest import TestCase
from elphem.const.unit import Length
from elphem.lattice.lattice import Lattice
from elphem.electron.free import FreeElectron

class TestUnit(TestCase):
    def setUp(self) -> None:
        a = 2.98 * Length.ANGSTROM["->"]
        lattice = Lattice('bcc', a, 'Li')

        self.electron = FreeElectron(lattice, n_band=20, n_electron=4)

    def test_band_structure(self):
        k_names = ["G", "H", "N", "G", "P", "H"]
        x, eig, x_special = self.electron.get_band_structure(k_names, n_split=20)
        
        self.assertEqual(eig.shape, (self.electron.n_band, len(x)))
        self.assertEqual(len(k_names), len(x_special))
    
    def test_get_reciprocal_vector(self):
        g = self.electron.reciprocal_vectors
        
        self.assertEqual(g.shape, (self.electron.n_band,3))