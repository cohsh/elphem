from unittest import TestCase
from elphem.const.atomic_weight import *

class TestUnit(TestCase):
    def test_atomic_weight(self):
        self.assertEqual(AtomicWeight.table["C"], 12.011)

        atom_name_list = ["Ga", "As"]
        weight_list = AtomicWeight.get_from_list(atom_name_list)
        ref_list = [AtomicWeight.table[s] for s in atom_name_list]
        self.assertEqual(weight_list, ref_list)