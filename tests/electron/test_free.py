from unittest import TestCase
from elphem.lattice.empty import EmptyLattice
from elphem.electron.free import FreeElectron

class TestUnit(TestCase):
    def setUp(self) -> None:
        lattice = EmptyLattice('fcc', 5.0)
        n_band = 20
        n_electron = 4
        self.electron = FreeElectron(lattice, n_band, n_electron)

    def test_band_structure(self):
        k_names = ["L", "G", "X"]        
        x, eig, x_special = self.electron.get_band_structure(k_names, n_split=20)
        
        self.assertEqual(eig.shape, (self.electron.n_band, len(x)))
        self.assertEqual(len(k_names), len(x_special))
    
    def test_get_reciprocal_vector(self):
        g = self.electron.get_reciprocal_vector()
        
        self.assertEqual(g.shape, (self.electron.n_band,3))