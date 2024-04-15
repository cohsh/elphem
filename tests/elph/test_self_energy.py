import numpy as np
from unittest import TestCase

from elphem.const.unit import Mass
from elphem.lattice.empty import EmptyLattice
from elphem.electron.free import FreeElectron
from elphem.phonon.debye import DebyeModel
from elphem.elph.self_energy import SelfEnergy

class TestUnit(TestCase):
    def test_calculate(self):
        lattice = EmptyLattice('fcc', 5.0)

        n_band = 10
        electron = FreeElectron(lattice, n_band, 4)
        
        mass = 12 * Mass.DALTON["->"]
        
        debye_temperature = 2300.0

        phonon = DebyeModel(lattice, debye_temperature, 2, mass)

        temperature = debye_temperature

        self_energy = SelfEnergy(lattice, electron, phonon, temperature)

        n_k = np.array([5,5,5])
        n_q = np.array([5,5,5])
        
        g_mesh, k_mesh = electron.grid(n_k)
        
        g = g_mesh.reshape(-1, 3)
        k = k_mesh.reshape(-1, 3)
        
        shape_mesh = g_mesh[..., 0].shape

        fan_term = np.array([self_energy.calculate_fan_term(g_i, k_i, n_q) for g_i, k_i in zip(g, k)]).reshape(shape_mesh)
        qp_strength = np.array([self_energy.calculate_qp_strength(g_i, k_i, n_q) for g_i, k_i in zip(g, k)]).reshape(shape_mesh)
        
        correct_shape = (n_band, np.prod(n_k))
        
        for v in [fan_term, qp_strength]:
            self.assertEqual(v.shape, correct_shape)