import numpy as np
from unittest import TestCase

from elphem.common.unit import Length
from elphem.lattice.lattice import Lattice
from elphem.electron.free import FreeElectron
from elphem.phonon.debye import DebyePhonon
from elphem.elph.electron_phonon import ElectronPhonon
from elphem.elph.spectrum import Spectrum, SpectrumBW

class TestUnit(TestCase):
    def setUp(self) -> None:
        a = 2.98 * Length.ANGSTROM["->"]
        debye_temperature = 344.0
        n_band = 8
        n_q = np.full(3, 5)

        self.range_omega = [-1.0, 2.0]

        temperature = 0.3 * debye_temperature

        lattice = Lattice('bcc', 'Li', a)
        electron = FreeElectron(lattice, n_band, 1)
        phonon = DebyePhonon(lattice, temperature)

        electron_phonon = ElectronPhonon(electron, phonon, temperature, n_q)
        self.spectrum = Spectrum(electron_phonon)
        self.spectrum_bw = SpectrumBW(electron_phonon)

    def test_calculate_with_grid(self):
        n_k = np.full(3, 5)
        n_omega = 100
        
        a = self.spectrum.get_with_grid(n_k, n_omega, self.range_omega)

        self.assertEqual(a.shape, (np.prod(n_k), n_omega))
    
    def test_calculate_with_path(self):
        k_names = ["G", "H", "N", "G", "P", "H"]
        n_split = 20
        
        n_omega = 200
        
        k, omegas, a, special_k = self.spectrum.get_with_path(k_names, n_split, n_omega, self.range_omega)
        
        self.assertEqual(a.shape, (len(k), len(omegas)))
        self.assertEqual(len(k_names), len(special_k))

    def test_bw_with_path(self):
        k_names = ["G", "H"]
        n_split = 20
        
        n_omega = 20
        
        k, omegas, a, special_k = self.spectrum_bw.get_with_path(k_names, n_split, n_omega, self.range_omega)
        
        self.assertEqual(a.shape, (len(k), len(omegas)))
        self.assertEqual(len(k_names), len(special_k))