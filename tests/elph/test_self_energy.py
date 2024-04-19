import numpy as np
from unittest import TestCase

from elphem.const.unit import Length
from elphem.lattice.lattice import Lattice
from elphem.electron.free import FreeElectron
from elphem.phonon.debye import DebyePhonon
from elphem.elph.electron_phonon import ElectronPhonon

class TestUnit(TestCase):
    def setUp(self) -> None:
        self.n_band = 10
        
        a = 2.98 * Length.ANGSTROM["->"]
        lattice = Lattice('bcc', 'Li', a)

        debye_temperature = 344.0
        temperature = 0.3 * debye_temperature
        
        phonon = DebyePhonon(lattice, debye_temperature)

        self.electron = FreeElectron(lattice, self.n_band, 4)
        self.electron_phonon = ElectronPhonon(self.electron, phonon, temperature)

    def test_calc(self):
        n_k = np.array([5,5,5])
        n_q = np.array([5,5,5])
        
        g_mesh, k_mesh = self.electron.get_gk_grid(n_k)
        
        g = g_mesh.reshape(-1, 3)
        k = k_mesh.reshape(-1, 3)
        
        shape_mesh = g_mesh[..., 0].shape

        self_energy = np.array([self.electron_phonon.get_self_energy(g_i, k_i, n_q) for g_i, k_i in zip(g, k)]).reshape(shape_mesh)
        coupling_strength = np.array([self.electron_phonon.get_coupling_strength(g_i, k_i, n_q) for g_i, k_i in zip(g, k)]).reshape(shape_mesh)
        qp_strength = np.array([self.electron_phonon.get_qp_strength(g_i, k_i, n_q) for g_i, k_i in zip(g, k)]).reshape(shape_mesh)
        
        correct_shape = (self.n_band, np.prod(n_k))
        
        for v in [self_energy, coupling_strength, qp_strength]:
            self.assertEqual(v.shape, correct_shape)