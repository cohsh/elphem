from dataclasses import dataclass

from elphem.lattice.empty import EmptyLattice
from elphem.electron.free import FreeElectron
from elphem.phonon.debye import DebyeModel

@dataclass
class Strength:
    lattice: EmptyLattice
    electron: FreeElectron
    phonon: DebyeModel
    sigma: float = 0.01
    effective_potential: float = 1 / 16