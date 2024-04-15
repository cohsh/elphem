import numpy as np
import matplotlib.pyplot as plt

from elphem import SpecialPoints, Energy, Length, LatticeConstant, EmptyLattice, FreeElectron

def main():
    # Example: Li (BCC)
    a = 2.98 * Length.ANGSTROM["->"]
    n_band = 50

    lattice = EmptyLattice('bcc', a)
    electron = FreeElectron(lattice, n_band, 1)
        
    k_names = ["G", "H", "N", "G", "P", "H"]
    n_split = 20

    x, eig, x_special = electron.get_band_structure(k_names, n_split)

    fig, ax = plt.subplots()
    for band in eig:
        ax.plot(x, band * Energy.EV["<-"], color="tab:blue")
    
    ax.vlines(x_special, ymin=-10, ymax=50, color="black", linewidth=0.3)
    ax.set_xticks(x_special)
    ax.set_xticklabels(k_names)
    ax.set_ylabel("Energy ($\mathrm{eV}$)")
    ax.set_ylim([-10,50])

    fig.savefig("test_band_structure.png")

if __name__ == "__main__":
    main()