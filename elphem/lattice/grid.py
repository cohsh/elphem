from dataclasses import dataclass

from elphem.lattice.empty import EmptyLattice

@dataclass
class Grid:
    lattice: EmptyLattice

class SimpleGrid(Grid):
    pass

class MonkhorstPackGrid(Grid):
    pass