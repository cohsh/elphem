from unittest import TestCase
import numpy as np
from elphem.const.unit import Mass
from elphem.const.atomic_weight import AtomicWeight
from elphem.lattice.empty import EmptyLattice
from elphem.phonon.debye import DebyeModel

class TestUnit(TestCase):
    def setUp(self) -> None:
        # Example: FCC-Fe
        lattice = EmptyLattice('fcc', 2.58)
        self.phonon = DebyeModel(lattice, 470.0, 1, AtomicWeight.table["Fe"] * Mass.DALTON["->"])

    def test_grid(self):
        nq = np.full(3,8)
        q = self.phonon.grid(nq)
        omega = self.phonon.eigenenergy(q)
        
        self.assertEqual(omega.shape, (np.prod(nq),))
    
    def test_dispersion(self):
        q_names = ["L", "G", "X"]
        x, omega, x_special = self.phonon.get_dispersion(q_names, n_split=20)

        self.assertEqual(len(omega), len(x))
        self.assertEqual(len(q_names), len(x_special))