from unittest import TestCase
import numpy as np

from elphem.const.brillouin import SpecialPoints
from elphem.lattice.empty import EmptyLattice
from elphem.electron.free import FreeElectron

class TestUnit(TestCase):
    def test_band_structure(self):
        lattice = EmptyLattice('fcc', 5.0)
        n_band = 20
        n_electron = 4
        electron = FreeElectron(lattice, n_band, n_electron)
            
        k_names = ["L", "G", "X"]
        k_via = []

        for name in k_names:
            k_via.append(SpecialPoints.FCC[name])
        
        x, eig, x_special = electron.get_band_structure(*k_via)
        
        self.assertEqual(len(eig), n_band)
        self.assertEqual(len(eig[0]), len(x))
        self.assertEqual(len(k_names), len(x_special))
    
    def test_get_reciprocal_vector(self):
        lattice = EmptyLattice('fcc', 5.0)
        n_band = 20
        n_electron = 4
        electron = FreeElectron(lattice, n_band, n_electron)

        g = electron.get_reciprocal_vector()
        
        self.assertEqual(len(g), n_band)
        self.assertEqual(len(g[0]), 3)