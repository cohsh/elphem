from unittest import TestCase
import numpy as np
from elphem.const.unit import Mass
from elphem.const.atomic_weight import AtomicWeight
from elphem.lattice.lattice import Lattice
from elphem.phonon.debye import DebyePhonon

class TestUnit(TestCase):
    def setUp(self) -> None:
        # Example: FCC-Fe
        lattice = Lattice('fcc', 2.58)
        self.phonon = DebyePhonon(lattice, 470.0, 1, AtomicWeight.table["Fe"] * Mass.DALTON["->"])
    
    def test_dispersion(self):
        q_names = ["L", "G", "X"]
        x, omega, x_special = self.phonon.get_dispersion(q_names, n_split=20)

        self.assertEqual(len(omega), len(x))
        self.assertEqual(len(q_names), len(x_special))