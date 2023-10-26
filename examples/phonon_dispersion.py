import os
import matplotlib.pyplot as plt

from elphem import LatticeConstant, EmptyLattice, DebyeModel, AtomicWeight, Energy, Mass, Length, SpecialPoints, Prefix

def main():
    # Example: \gamma-Fe (FCC)
    a = 2.58 * Length.ANGSTROM["->"]
    lattice_constant = LatticeConstant(a, a, a, 60, 60, 60)
    lattice = EmptyLattice(lattice_constant)
    
    debye_temperature = 470.0

    phonon = DebyeModel(lattice, debye_temperature, 1, AtomicWeight.table["Fe"] * Mass.DALTON["->"])

    q_names = ["G", "X", "G", "L"]
    q_via = [SpecialPoints.FCC[name] for name in q_names]
    
    x, omega, x_special = phonon.get_dispersion(*q_via)
    
    fig, ax = plt.subplots()

    ax.plot(x, omega * Energy.EV["<-"] / Prefix.MILLI, color="tab:blue")
    
    for x0 in x_special:
        ax.axvline(x=x0, color="black", linewidth=0.3)
    
    ax.set_xticks(x_special)
    ax.set_xticklabels(q_names)
    ax.set_ylabel("Energy ($\mathrm{meV}$)")

    fig.savefig("test_phonon_dispersion.png")

if __name__ == "__main__":
    main()