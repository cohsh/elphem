from unittest import TestCase
import numpy as np

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

        phonon = DebyeModel(lattice, 470.0, 1, AtomicWeight.table["Fe"] * Mass.DALTON["->"])
        
        nq = np.array([8]*3)
        q = phonon.grid(nq)
        omega = phonon.eigenenergy(q)
        
        self.assertEqual(omega.shape, (nq[0], nq[1], nq[2]))
    
    def test_get_dispersion(self):
        # Example: FCC-Fe
        lattice_constant = LatticeConstant(2.58, 2.58, 2.58, 60, 60, 60)
        lattice = EmptyLattice(lattice_constant)

        phonon = DebyeModel(lattice, 470.0, 1, AtomicWeight.table["Fe"] * Mass.DALTON["->"])

        q_names = ["L", "G", "X"]
        q_via = []
        
        for name in q_names:
            q_via.append(SpecialPoints.FCC[name])
        
        x, omega, x_special = phonon.get_dispersion(*q_via)

        self.assertEqual(len(omega), len(x))
        self.assertEqual(len(q_names), len(x_special))