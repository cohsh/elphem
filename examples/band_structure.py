"""Example: bcc-Li"""
import matplotlib.pyplot as plt
from elphem import *

def main():
    a = 2.98 * Length.ANGSTROM['->']
    n_electrons = 1
    n_bands = 20

    lattice = Lattice3D('bcc', 'Li', a)
    k_names = ["G", "H", "N", "G", "P", "H"]
    
    k_path = lattice.reciprocal.get_path(k_names, 100)

    electron = Electron.create_from_path(lattice, n_electrons, n_bands, k_path)

    eigenenergies = electron.eigenenergies * Energy.EV['<-']

    fig, ax = plt.subplots()
    for band in eigenenergies:
        ax.plot(k_path.minor_scales, band, color="tab:blue")
    
    y_range = [-10, 50]
    
    ax.vlines(k_path.major_scales, ymin=y_range[0], ymax=y_range[1], color="black", linewidth=0.3)
    ax.set_xticks(k_path.major_scales)
    ax.set_xticklabels(k_names)
    ax.set_ylabel("Energy ($\mathrm{eV}$)")
    ax.set_ylim(y_range)

    fig.savefig("band_structure.png")

if __name__ == "__main__":
    main()