from unittest import TestCase
import numpy as np

from elphem.lattice.empty import EmptyLattice

class TestUnit(TestCase):
    def test_vector(self):
        lattice = EmptyLattice(5, 5, 5, 60, 60, 60)
        basis_primitive = lattice.basis["primitive"]
        basis_reciprocal = lattice.basis["reciprocal"]

        self.assertEqual(basis_primitive.shape, (3,3))        
        self.assertEqual(basis_reciprocal.shape, (3,3))
    
    def test_volume(self):
        lattice = EmptyLattice(4.65, 4.65, 4.65, 90, 90, 90)
        volume = lattice.volume["primitive"]
        self.assertTrue(abs(volume - np.prod(lattice.constants.length)) < 1e-10)