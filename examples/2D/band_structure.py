"""Example: bcc-Li"""
import matplotlib.pyplot as plt
from elphem import *

def main():
    a = 2.98 * Length.ANGSTROM['->']

    lattice = Lattice2D('square', 'Li', a)
    k_names = ["G", "X", "G", "M"]
    
    k_path = lattice.reciprocal.get_path(k_names, 40)

    electron = FreeElectron.create_from_path(lattice, 1, 2, k_path)

    eigenenergies = electron.eigenenergies * Energy.EV['<-']

    fig, ax = plt.subplots()
    for band in eigenenergies:
        ax.plot(k_path.minor_scales, band, color="tab:blue")
    
    ax.vlines(k_path.major_scales, ymin=np.amin(eigenenergies), ymax=np.amax(eigenenergies), color="black", linewidth=0.3)
    ax.set_xticks(k_path.major_scales)
    ax.set_xticklabels(k_names)
    ax.set_ylabel("Energy ($\mathrm{eV}$)")

    fig.savefig("band_structure.png")

if __name__ == "__main__":
    main()