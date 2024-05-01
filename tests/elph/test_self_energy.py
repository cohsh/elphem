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
        n_k = np.full(3,5)
        n_q = np.full(3,5)
        
        lattice = Lattice('bcc', 'Li', a)

        debye_temperature = 344.0
        temperature = 0.3 * debye_temperature

        electron = FreeElectron.create_from_n(lattice, 1, self.n_band, n_k)        
        phonon = DebyePhonon.create_from_n(lattice, debye_temperature, n_q)
        self.electron_phonon = ElectronPhonon(electron, phonon)