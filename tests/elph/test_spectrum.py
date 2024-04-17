import numpy as np
from unittest import TestCase

from elphem.const.unit import Mass, Length
from elphem.const.atomic_weight import AtomicWeight
from elphem.lattice.empty import EmptyLattice
from elphem.electron.free import FreeElectron
from elphem.phonon.debye import DebyeModel
from elphem.elph.self_energy import SelfEnergy
from elphem.elph.spectrum import Spectrum

class TestUnit(TestCase):
    def setUp(self) -> None:
        a = 2.98 * Length.ANGSTROM["->"]
        mass = AtomicWeight.table["Li"] * Mass.DALTON["->"]
        debye_temperature = 344.0
        n_band = 8

        temperature = 0.3 * debye_temperature

        lattice = EmptyLattice('bcc', a)

        electron = FreeElectron(lattice, n_band, 1)
        phonon = DebyeModel(lattice, temperature, 1, mass)

        self_energy = SelfEnergy(lattice, electron, phonon, temperature)
        self.spectrum = Spectrum(self_energy)

    def test_calculate_with_grid(self):
        n_k = np.full(3, 5)        
        n_q = np.full(3, 5)
        n_omega = 100
        
        a = self.spectrum.calculate_with_grid(n_k, n_q, n_omega)

        self.assertEqual(a.shape, (np.prod(n_k), n_omega))
    
    def test_calculate_with_path(self):
        k_names = ["G", "H", "N", "G", "P", "H"]
        n_split = 20
        
        n_q = np.array([5]*3)
        n_omega = 200
        range_omega = [-1.0, 2.0]
        
        k, omegas, a, special_k = self.spectrum.calculate_with_path(k_names, n_split, n_q, n_omega, range_omega)
        
        self.assertEqual(a.shape, (len(k), len(omegas)))
        self.assertEqual(len(k_names), len(special_k))