import numpy as np
from unittest import TestCase

from elphem.const.unit import Mass
from elphem.const.brillouin import SpecialPoints
from elphem.lattice.empty import EmptyLattice
from elphem.electron.free import FreeElectron
from elphem.phonon.debye import DebyeModel
from elphem.elph.self_energy import SelfEnergy
from elphem.elph.spectrum import Spectrum

class TestUnit(TestCase):
    def test_calculate_with_grid(self):
        mass = 12 * Mass.DALTON["->"]        
        temperature = 2300.0

        lattice = EmptyLattice('fcc', 5.0)

        n_band = 10

        electron = FreeElectron(lattice, n_band, 4)        
        phonon = DebyeModel(lattice, temperature, 2, mass)

        self_energy = SelfEnergy(lattice, electron, phonon, temperature)

        n_k = np.array([5]*3)        
        n_q = np.array([5]*3)
        n_omega = 100
        
        spectrum = Spectrum(self_energy).calculate_with_grid(n_k, n_q, n_omega)
                
        self.assertEqual(spectrum.shape, (np.prod(n_k), n_omega))
    
    def test_calculate_with_path(self):
        mass = 12 * Mass.DALTON["->"]        
        temperature = 2300.0

        lattice = EmptyLattice('fcc', 5)

        n_band = 10

        electron = FreeElectron(lattice, n_band, 4)        
        phonon = DebyeModel(lattice, temperature, 2, mass)

        self_energy = SelfEnergy(lattice, electron, phonon, temperature)

        k_names = ["G", "H", "N", "G", "P", "H"]
        k_via = [SpecialPoints.BCC[name] for name in k_names]
        n_via = 20

        n_q = np.array([5]*3)
        n_omega = 200
        range_omega = [-1.0, 2.0]
        
        x, omegas, spectrum, special_x = Spectrum(self_energy).calculate_with_path(k_via, n_via, n_q, n_omega, range_omega)