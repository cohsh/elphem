import os

from elphem import LatticeConstant, EmptyLattice, DebyeModel, AtomicWeight, Mass, Length, FCC

def main():
    # Example: \gamma-Fe (FCC)
    a = 2.58 * Length.angstrom["->"]
    lattice_constant = LatticeConstant(a, a, a, 60, 60, 60)
    lattice = EmptyLattice(lattice_constant)
    
    debye_temperature = 470.0

    phonon = DebyeModel(lattice, debye_temperature, 1, AtomicWeight.table["Fe"] * Mass.Dalton["->"])

    q_names = ["G", "X", "G", "L"]
    q_via = []
    for name in q_names:
        q_via.append(FCC.points[name])
    
    file_name = "test_phonon_dispersion.png"
    phonon.save_dispersion(file_name, q_names, *q_via)

if __name__ == "__main__":
    main()