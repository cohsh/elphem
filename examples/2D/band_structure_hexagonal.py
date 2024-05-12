"""Example: bcc-Li"""
import matplotlib.pyplot as plt
from elphem import *

def main():
    a = 2.98 * Length.ANGSTROM['->']

    lattice = Lattice2D('hexagonal', 'C', a)
    k_names = ["M", "G", "K", "M"]
    
    k_path = lattice.reciprocal.get_path(k_names, 40)

    electron = FreeElectron.create_from_path(lattice, 4, 8, k_path)

    eigenenergies = electron.eigenenergies * Energy.EV['<-']

    fig, ax = plt.subplots()
    for band in eigenenergies:
        ax.plot(k_path.minor_scales, band, color="tab:blue")
    
    ax.vlines(k_path.major_scales, ymin=np.amin(eigenenergies), ymax=np.amax(eigenenergies), color="black", linewidth=0.3)
    ax.set_xticks(k_path.major_scales)
    ax.set_xticklabels(k_names)
    ax.set_ylabel("Energy ($\mathrm{eV}$)")

    fig.savefig("band_structure_hexagonal.png")

if __name__ == "__main__":
    main()