from unittest import TestCase
import numpy as np

from elphem.const.unit import Length
from elphem.lattice.lattice import *

class TestUnit(TestCase):
    def setUp(self) -> None:
        a = 2.98 * Length.ANGSTROM["->"]
        self.lattice = Lattice('bcc', 'Li', a)
        self.basis_primitive = self.lattice.basis["primitive"]
        self.basis_reciprocal = self.lattice.basis["reciprocal"]

    def test_vector(self):
        for b in [self.basis_primitive, self.basis_reciprocal]:
            self.assertEqual(b.shape, (3,3))

    def test_grid(self):
        n = [8, 8, 8]

        grid = self.lattice.reciprocal_cell.get_monkhorst_pack_grid(*n)
        
        correct_shape = (np.prod(n), 3)
        
        self.assertEqual(grid.shape, correct_shape)