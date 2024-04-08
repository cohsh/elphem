from dataclasses import dataclass

from elphem.elph.self_energy import SelfEnergy2nd

@dataclass
class Spectrum:
    self_energy: SelfEnergy2nd