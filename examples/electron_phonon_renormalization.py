"""Example: bcc-Li"""
import numpy as np
import matplotlib.pyplot as plt
from elphem import *

def main():
    a = 2.98 * Length.ANGSTROM["->"]
    debye_temperature = 344.0
    temperature = 3 * debye_temperature
    n_band = 20
    n_electron = 1

    lattice = Lattice('bcc', a, 'Li')
    electron = FreeElectron(lattice, n_band, n_electron)        
    phonon = DebyePhonon(lattice, temperature)

    electron_phonon = ElectronPhonon(electron, phonon, temperature)

    k_names = ["G", "H", "N", "G", "P", "H"]

    n_split = 20
    n_q = np.full(3, 12)
    
    k, eig, epr, special_k = EPR(electron_phonon).get_with_path(k_names, n_split, n_q)
    
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


    fig.savefig("example_epr.png")

if __name__ == "__main__":
    main()