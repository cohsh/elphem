import numpy as np
from unittest import TestCase

from elphem.common.unit import Length
from elphem.lattice.lattice import Lattice
from elphem.electron.free import FreeElectron
from elphem.phonon.debye import DebyePhonon
from elphem.elph.electron_phonon import ElectronPhonon

class TestUnit(TestCase):
    def setUp(self) -> None:
        self.n_band = 10
        a = 2.98 * Length.ANGSTROM["->"]
        n_q = np.full(3,5)
        
        lattice = Lattice('bcc', 'Li', a)

        debye_temperature = 344.0
        temperature = 0.3 * debye_temperature
        
        phonon = DebyePhonon(lattice, debye_temperature)

        self.electron = FreeElectron(lattice, self.n_band, 1)
        self.electron_phonon = ElectronPhonon(self.electron, phonon, temperature, n_q)