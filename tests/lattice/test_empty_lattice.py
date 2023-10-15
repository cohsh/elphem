from unittest import TestCase
import numpy as np

from elphem.lattice.empty import EmptyLattice, LatticeConstant

class TestUnit(TestCase):
    def test_vector(self):
        lattice_constant = LatticeConstant(5, 5, 5, 60, 60, 60)
        lattice = EmptyLattice(lattice_constant)
        basis_primitive = lattice.basis["primitive"]
        basis_reciprocal = lattice.basis["reciprocal"]

        self.assertEqual(basis_primitive.shape, (3,3))        
        self.assertEqual(basis_reciprocal.shape, (3,3))
    
    def test_volume(self):
        lattice_constant = LatticeConstant(4.65, 4.65, 4.65, 90, 90, 90)
        lattice = EmptyLattice(lattice_constant)
        volume = lattice.volume["primitive"]
        self.assertTrue(abs(volume - np.prod(lattice_constant.length)) < 1e-10)