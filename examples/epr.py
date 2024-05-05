"""Example: bcc-Li"""
import numpy as np
import matplotlib.pyplot as plt
from elphem import *

def main():
    a = 2.98 * Length.ANGSTROM["->"]
    debye_temperature = 344.0
    n_band = 4
    n_electron = 1
    n_q = np.full(3, 12)
    k_names = ["G", "H", "N", "G", "P", "H"]
    n_split = 20

    lattice = Lattice('bcc', 'Li', a, debye_temperature)
    k_path = lattice.reciprocal.get_path(k_names, n_split)

    electron = FreeElectron.create_from_path(lattice, n_electron, n_band, k_path)
    phonon = DebyePhonon.create_from_n(lattice, debye_temperature, n_q)

    electron_phonon = ElectronPhonon(electron, phonon)
    
    for i_g in n_band:
        for i_k in electron.n_k:
            pass
    
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for n in range(n_band):
        if n == 0:
            ax.plot(k, (eig[n] + epr[n]) * Energy.EV["<-"], color="tab:orange", label="w/ EPR")
            ax.plot(k, eig[n] * Energy.EV["<-"], color="tab:blue", label="w/o EPR")
        else:
            ax.plot(k, (eig[n] + epr[n]) * Energy.EV["<-"], color="tab:orange")
            ax.plot(k, eig[n] * Energy.EV["<-"], color="tab:blue")

    for k0 in special_k:
        ax.axvline(x=k0, color="black", linewidth=0.3)
    
    ax.set_xticks(special_k)
    ax.set_xticklabels(k_names)
    ax.set_ylabel("Energy ($\mathrm{eV}$)")
    ax.set_title("Example: Band structure of bcc-Li")
    ax.set_ylim([-10,20])
    ax.legend()


    fig.savefig("epr.png")

if __name__ == "__main__":
    main()