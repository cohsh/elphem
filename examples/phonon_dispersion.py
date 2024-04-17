import matplotlib.pyplot as plt
from elphem import *

def main():
    # Example: bcc-Li
    a = 2.98 * Length.ANGSTROM["->"]
    mass = AtomicWeight.table["Li"] * Mass.DALTON["->"]
    lattice = EmptyLattice('bcc', a)

    debye_temperature = 344.0
    phonon = DebyeModel(lattice, debye_temperature, 1, mass)

    q_names = ["G", "H", "N", "G", "P", "H"]
    
    x, omega, x_special = phonon.get_dispersion(q_names, n_split=20)
    
    fig, ax = plt.subplots()

    ax.plot(x, omega * Energy.EV["<-"] * 1.0e+3, color="tab:blue")
    
    for x0 in x_special:
        ax.axvline(x=x0, color="black", linewidth=0.3)
    
    ax.set_xticks(x_special)
    ax.set_xticklabels(q_names)
    ax.set_ylabel("Energy ($\mathrm{meV}$)")

    fig.savefig("test_phonon_dispersion.png")

if __name__ == "__main__":
    main()