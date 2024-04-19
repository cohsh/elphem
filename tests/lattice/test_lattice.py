from unittest import TestCase
import numpy as np

from elphem.const.unit import Length
from elphem.lattice.lattice import *
from elphem.lattice.grid import *

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
        grid = Grid(self.basis_reciprocal, *n)
        simple_grid = SimpleGrid(self.basis_reciprocal, *n)
        mk_grid = MonkhorstPackGrid(self.basis_reciprocal, *n)
        
        correct_shape = (np.prod(n), 3)
        
        for g in [grid, simple_grid, mk_grid]:
            self.assertEqual(np.prod(n), g.n_mesh)
            self.assertEqual(g.mesh.shape, correct_shape)