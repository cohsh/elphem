"""Example: bcc-Li"""
import matplotlib.pyplot as plt
from elphem import *

def main():
    a = 2.98 * Length.ANGSTROM["->"]

    lattice = EmptyLattice('bcc', a)
    electron = FreeElectron(lattice, n_band=50, n_electron=1)

    k_names = ["G", "H", "N", "G", "P", "H"]

    k, eig, special_k = electron.get_band_structure(k_names, n_split=20)

    fig, ax = plt.subplots()
    for band in eig:
        ax.plot(k, band * Energy.EV["<-"], color="tab:blue")
    
    ax.vlines(special_k, ymin=-10, ymax=50, color="black", linewidth=0.3)
    ax.set_xticks(special_k)
    ax.set_xticklabels(k_names)
    ax.set_ylabel("Energy ($\mathrm{eV}$)")
    ax.set_ylim([-10,50])

    fig.savefig("example_band_structure.png")

if __name__ == "__main__":
    main()