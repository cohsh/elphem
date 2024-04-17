"""Example: bcc-Li"""
import numpy as np
import matplotlib.pyplot as plt
from elphem import *

def main():
    a = 2.98 * Length.ANGSTROM["->"]
    mass = AtomicWeight.table["Li"] * Mass.DALTON["->"]
    debye_temperature = 344.0
    temperature = 3 * debye_temperature
    n_band = 20

    lattice = EmptyLattice('bcc', a)
    electron = FreeElectron(lattice, n_band, 1)        
    phonon = DebyeModel(lattice, temperature, 1, mass)

    self_energy = SelfEnergy(lattice, electron, phonon, temperature, eta=0.05)

    k_names = ["G", "H", "N", "G", "P", "H"]

    n_split = 20
    n_q = np.array([8]*3)
    
    k, eig, epr, special_k = EPR(self_energy).calculate_with_path(k_names, n_split, n_q)
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    for n in range(n_band):
        if n == 0:
            ax.plot(k, eig[n] * Energy.EV["<-"], color="tab:blue", label="w/o EPR")
            ax.plot(k, (eig[n] + epr[n]) * Energy.EV["<-"], color="tab:orange", label="w/ EPR")
        else:
            ax.plot(k, eig[n] * Energy.EV["<-"], color="tab:blue")
            ax.plot(k, (eig[n] + epr[n]) * Energy.EV["<-"], color="tab:orange")
    
    for k0 in special_k:
        ax.axvline(x=k0, color="black", linewidth=0.3)
    
    ax.set_xticks(special_k)
    ax.set_xticklabels(k_names)
    ax.set_ylabel("Energy ($\mathrm{eV}$)")
    ax.set_title("Example: Band structure of bcc-Li")
    ax.set_ylim([-10,20])
    ax.legend()


    fig.savefig("example_epr.png")

if __name__ == "__main__":
    main()