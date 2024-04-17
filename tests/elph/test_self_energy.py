import numpy as np
from unittest import TestCase

from elphem.const.unit import Mass
from elphem.lattice.empty import EmptyLattice
from elphem.electron.free import FreeElectron
from elphem.phonon.debye import DebyeModel
from elphem.elph.self_energy import SelfEnergy

class TestUnit(TestCase):
    def setUp(self) -> None:
        self.n_band = 10
        
        mass = 12 * Mass.DALTON["->"]
        debye_temperature = 2300.0
        temperature = 0.3 * debye_temperature
        
        lattice = EmptyLattice('fcc', 5.0)
        phonon = DebyeModel(lattice, debye_temperature, 2, mass)

        self.electron = FreeElectron(lattice, self.n_band, 4)
        self.self_energy = SelfEnergy(lattice, self.electron, phonon, temperature)

    def test_calc(self):
        n_k = np.array([5,5,5])
        n_q = np.array([5,5,5])
        
        g_mesh, k_mesh = self.electron.grid(n_k)
        
        g = g_mesh.reshape(-1, 3)
        k = k_mesh.reshape(-1, 3)
        
        shape_mesh = g_mesh[..., 0].shape

        fan_term = np.array([self.self_energy.calculate_fan_term(g_i, k_i, n_q) for g_i, k_i in zip(g, k)]).reshape(shape_mesh)
        coupling_strength = np.array([self.self_energy.calculate_coupling_strength(g_i, k_i, n_q) for g_i, k_i in zip(g, k)]).reshape(shape_mesh)
        qp_strength = np.array([self.self_energy.calculate_qp_strength(g_i, k_i, n_q) for g_i, k_i in zip(g, k)]).reshape(shape_mesh)
        
        correct_shape = (self.n_band, np.prod(n_k))
        
        for v in [fan_term, coupling_strength, qp_strength]:
            self.assertEqual(v.shape, correct_shape)