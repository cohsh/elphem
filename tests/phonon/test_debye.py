from unittest import TestCase
import numpy as np
import os
import random
import string

from elphem.const.unit import Mass
from elphem.const.brillouin import SpecialPoints
from elphem.const.atomic_weight import AtomicWeight
from elphem.lattice.empty import EmptyLattice, LatticeConstant
from elphem.phonon.debye import DebyeModel

class TestUnit(TestCase):
    def test_debye(self):
        # Example: FCC-Fe
        lattice_constant = LatticeConstant(2.58, 2.58, 2.58, 60, 60, 60)
        lattice = EmptyLattice(lattice_constant)

        phonon = DebyeModel(lattice, 470.0, 1, AtomicWeight.table["Fe"] * Mass.Dalton["->"])
        
        nq = np.array([8]*3)
        q = phonon.grid(nq)
        omega = phonon.eigenenergy(q)
        
        self.assertEqual(omega.shape, (nq[0], nq[1], nq[2]))
    
    def test_save_figure(self):
        # Example: FCC-Fe
        lattice_constant = LatticeConstant(2.58, 2.58, 2.58, 60, 60, 60)
        lattice = EmptyLattice(lattice_constant)

        phonon = DebyeModel(lattice, 470.0, 1, AtomicWeight.table["Fe"] * Mass.Dalton["->"])

        q_names = ["L", "G", "X"]
        q_via = []
        for name in q_names:
            q_via.append(SpecialPoints.FCC[name])
        
        file_name = "".join(random.choices(string.ascii_letters + string.digits, k=20)) + ".png"
        phonon.save_dispersion(file_name, q_names, *q_via)
        is_file = os.path.isfile(file_name)
        if is_file:
            os.remove(file_name)