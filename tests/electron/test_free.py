from unittest import TestCase
import numpy as np
import os
import random
import string

from elphem.const.brillouin import SpecialPoints
from elphem.lattice.empty import LatticeConstant, EmptyLattice
from elphem.electron.free import FreeElectron

class TestUnit(TestCase):
    def test_figure(self):
        lattice_constant = LatticeConstant(5,5,5,60,60,60)
        lattice = EmptyLattice(lattice_constant)
        n_cut = np.array([2]*3)
        electron = FreeElectron(lattice, 4)
            
        k_names = ["L", "G", "X"]
        k_via = []

        for name in k_names:
            k_via.append(SpecialPoints.FCC[name])
        
        file_name = "".join(random.choices(string.ascii_letters + string.digits, k=20)) + ".png"
        
        electron.save_band(file_name, n_cut, k_names, *k_via)
        is_file = os.path.isfile(file_name)
        if is_file:
            os.remove(file_name)
        
        self.assertTrue(is_file)