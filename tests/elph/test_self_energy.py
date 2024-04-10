import numpy as np
from unittest import TestCase

from elphem.const.unit import Mass
from elphem.lattice.empty import EmptyLattice, LatticeConstant
from elphem.electron.free import FreeElectron
from elphem.phonon.debye import DebyeModel
from elphem.elph.self_energy import SelfEnergy

class TestUnit(TestCase):
    def test_calculate(self):
        lattice_constant = LatticeConstant(5,5,5,60,60,60)
        lattice = EmptyLattice(lattice_constant)

        electron = FreeElectron(lattice, 4)
        
        mass = 12 * Mass.DALTON["->"]
        
        debye_temperature = 2300.0

        phonon = DebyeModel(lattice, debye_temperature, 2, mass)

        temperature = debye_temperature

        self_energy = SelfEnergy(lattice, electron, phonon, temperature)

        n_g = np.array([1]*3)
        n_k = np.array([5]*3)
        g, k = electron.grid(n_g, n_k)
        
        n_g_inter = np.array([1]*3)
        n_q = np.array([5]*3)
        selfen = self_energy.calculate_fan_term(g, k, n_g_inter, n_q)
        
        self.assertEqual(selfen.shape, (n_g[0]*2, n_g[1]*2, n_g[2]*2) + (n_k[0], n_k[1], n_k[2]))