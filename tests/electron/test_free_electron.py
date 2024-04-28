from unittest import TestCase
from elphem.common.unit import Length
from elphem.lattice.lattice import Lattice
from elphem.electron.free import FreeElectron

class TestUnit(TestCase):
    def setUp(self) -> None:
        a = 2.98 * Length.ANGSTROM["->"]
        lattice = Lattice('bcc', 'Li', a)

        self.electron = FreeElectron(lattice, n_band=20, n_electron=4)

    def test_band_structure(self):
        k_names = ["G", "H", "N", "G", "P", "H"]
        eig_path = self.electron.get_band_structure(k_names, n_split=20)
        
        self.assertEqual(eig_path.values.shape, (self.electron.n_band, len(eig_path.distances)))
        self.assertEqual(len(k_names), len(eig_path.special_distances))
    
    def test_get_reciprocal_vector(self):
        g = self.electron.reciprocal_vectors
        
        self.assertEqual(g.shape, (self.electron.n_band,3))