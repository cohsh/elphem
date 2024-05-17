import numpy as np
from unittest import TestCase

from elphem.common.unit import Length
from elphem.lattice.lattice import Lattice3D
from elphem.electron.electron import Electron
from elphem.phonon.phonon import Phonon
from elphem.elph.electron_phonon import ElectronPhonon

class TestUnit(TestCase):
    def setUp(self) -> None:
        n_electrons = 1
        n_bands_electrons = 10
        n_bands_elph = 2
        a = 2.98 * Length.ANGSTROM["->"]
        n_k = np.full(3,4)
        n_q = np.full(3,4)
        
        lattice = Lattice3D('bcc', 'Li', a)

        debye_temperature = 344.0
        temperature = 0.3 * debye_temperature

        self.electron = Electron.create_from_n(lattice, n_electrons, n_bands_electrons, n_k)
        self.phonon = Phonon.create_from_n(lattice, debye_temperature, n_q)
        self.electron_phonon = ElectronPhonon(self.electron, self.phonon, temperature, n_bands_elph)
    
    def test_coupling(self):
        coupling2 = self.electron_phonon.coupling2
        
        self.assertEqual(coupling2.shape, (self.electron_phonon.n_bands, self.electron.n_bands, self.electron.n_k, self.phonon.n_q))
    
    def test_self_energies(self):
        self_energies = self.electron_phonon.calculate_self_energies(0.0)
        
        self.assertEqual(self_energies.shape, (self.electron_phonon.n_bands, self.electron.n_k))
    
    def test_spectrum(self):
        spectrum = self.electron_phonon.calculate_spectrum(0.0)
        
        self.assertEqual(spectrum.shape, (self.electron.n_k))
    
    def test_self_energies_over_range(self):
        omega_array = np.linspace(-1.0, 1.0, 50)
        self_energies = self.electron_phonon.calculate_self_energies_over_range(omega_array)
        
        self.assertEqual(self_energies.shape, (self.electron_phonon.n_bands, self.electron.n_k, len(omega_array)))
    
    def test_spectrum_over_range(self):
        omega_array = np.linspace(-1.0, 1.0, 50)
        self_energies = self.electron_phonon.calculate_spectrum_over_range(omega_array)
        
        self.assertEqual(self_energies.shape, (self.electron.n_k, len(omega_array)))