"""Example: bcc-Li"""
import matplotlib.pyplot as plt
from elphem import *

def main():
    a = 2.98 * Length.ANGSTROM["->"]
    mass = AtomicWeight.table["Li"] * Mass.DALTON["->"]
    lattice = EmptyLattice('bcc', a)

    debye_temperature = 344.0
    phonon = DebyeModel(lattice, debye_temperature, 1, mass)

    q_names = ["G", "H", "N", "G", "P", "H"]
    
    q, omega, special_q = phonon.get_dispersion(q_names, n_split=20)
    
    fig, ax = plt.subplots()

    ax.plot(q, omega * Energy.EV["<-"] * 1.0e+3, color="tab:blue")
    
    for q0 in special_q:
        ax.axvline(x=q0, color="black", linewidth=0.3)
    
    ax.set_xticks(special_q)
    ax.set_xticklabels(q_names)
    ax.set_ylabel("Energy ($\mathrm{meV}$)")

    fig.savefig("example_phonon_dispersion.png")

if __name__ == "__main__":
    main()