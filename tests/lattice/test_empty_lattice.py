from unittest import TestCase
import numpy as np

from elphem.const.unit import Length
from elphem.lattice.lattice import *
from elphem.lattice.grid import *

class TestUnit(TestCase):
    def setUp(self) -> None:
        a = 2.98 * Length.ANGSTROM["->"]
        self.lattice = Lattice('bcc', a, 'Li')

    def test_vector(self):
        basis_primitive = self.lattice.basis["primitive"]
        basis_reciprocal = self.lattice.basis["reciprocal"]

        for b in [basis_primitive, basis_reciprocal]:
            self.assertEqual(b.shape, (3,3))

    def test_grid(self):
        basis = self.lattice.basis["reciprocal"]

        n = [8, 8, 8]
        grid = Grid(basis, *n)
        simple_grid = SimpleGrid(basis, *n)
        mk_grid = MonkhorstPackGrid(basis, *n)
        
        correct_shape = (np.prod(n), 3)
        
        for g in [grid, simple_grid, mk_grid]:
            self.assertEqual(np.prod(n), g.n_mesh)
            self.assertEqual(g.mesh.shape, correct_shape)