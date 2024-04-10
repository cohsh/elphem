import numpy as np
from unittest import TestCase

from elphem.const.unit import Mass
from elphem.lattice.empty import EmptyLattice, LatticeConstant
from elphem.electron.free import FreeElectron
from elphem.phonon.debye import DebyeModel
from elphem.elph.self_energy import SelfEnergy
from elphem.elph.spectrum import Spectrum

class TestUnit(TestCase):
    def test_calculate(self):
        mass = 12 * Mass.DALTON["->"]        
        temperature = 2300.0

        lattice_constant = LatticeConstant(5,5,5,60,60,60)
        lattice = EmptyLattice(lattice_constant)

        electron = FreeElectron(lattice, 4)        
        phonon = DebyeModel(lattice, temperature, 2, mass)

        self_energy = SelfEnergy(lattice, electron, phonon, temperature)

        n_g = np.array([1]*3)
        n_k = np.array([5]*3)        
        n_g_inter = np.array([1]*3)
        n_q = np.array([5]*3)
        n_omega = 100
        
        a = Spectrum(self_energy).calculate(n_g, n_k, n_g_inter, n_q, n_omega)
        # print(a)
        
        self.assertEqual(a.shape, (n_g[0]*2, n_g[1]*2, n_g[2]*2) + (n_omega,))