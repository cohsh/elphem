from unittest import TestCase
import numpy as np

from elphem.lattice.lattice import *
from elphem.lattice.grid import *

class TestUnit(TestCase):
    def test_vector(self):
        lattice = Lattice('fcc', 5.0)
        basis_primitive = lattice.basis["primitive"]
        basis_reciprocal = lattice.basis["reciprocal"]

        for b in [basis_primitive, basis_reciprocal]:
            self.assertEqual(b.shape, (3,3))
    
    def test_volume(self):
        lattice = Lattice('sc', 4.65)
        volume = lattice.volume["primitive"]
        self.assertTrue(abs(volume - np.prod(lattice.constants.length)) < 1e-10)
    
    def test_grid(self):
        lattice = Lattice('bcc', 3.0)
        basis = lattice.basis["reciprocal"]

        n = [8, 8, 8]
        grid = Grid(basis, *n)
        simple_grid = SimpleGrid(basis, *n)
        mk_grid = MonkhorstPackGrid(basis, *n)
        
        correct_shape = (np.prod(n), 3)
        
        for g in [grid, simple_grid, mk_grid]:
            self.assertEqual(np.prod(n), g.n_mesh)
            self.assertEqual(g.mesh.shape, correct_shape)