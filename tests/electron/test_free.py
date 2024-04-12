from unittest import TestCase
import numpy as np

from elphem.const.brillouin import SpecialPoints
from elphem.lattice.empty import EmptyLattice
from elphem.electron.free import FreeElectron

class TestUnit(TestCase):
    def test_band_structure(self):
        lattice = EmptyLattice(5,5,5,60,60,60)
        n_cut = np.array([2]*3)
        electron = FreeElectron(lattice, 4)
            
        k_names = ["L", "G", "X"]
        k_via = []

        for name in k_names:
            k_via.append(SpecialPoints.FCC[name])
        
        x, eig, x_special = electron.get_band_structure(n_cut, *k_via)
        
        n_band = np.prod(n_cut) * 8
        self.assertEqual(len(eig), n_band)
        self.assertEqual(len(eig[0]), len(x))
        self.assertEqual(len(k_names), len(x_special))