"""Example: bcc-Li"""
import numpy as np
import matplotlib.pyplot as plt
from elphem import *

def main():
    a = 2.98 * Length.ANGSTROM["->"]
    debye_temperature = 344.0
    n_band = 1
    n_electron = 1
    n_q = np.full(1, 40)
    k_names = ["X", "G", "X"]
    n_split = 20

    lattice = Lattice1D('Li', a, debye_temperature * 2)
    k_path = lattice.reciprocal.get_path(k_names, n_split)

    electron = FreeElectron.create_from_path(lattice, n_electron, n_band, k_path)
    phonon = DebyePhonon.create_from_n(lattice, debye_temperature, n_q)

    electron_phonon = ElectronPhonon(electron, phonon, eta=0.001)
    
    eig = electron.eigenenergies
    epr = np.empty(electron.eigenenergies.shape)
    
    for i_g in range(n_band):
        for i_k in range(electron.n_k):
            epr[i_g, i_k] = electron_phonon.calculate_self_energies(eig[i_g, i_k])[i_g, i_k].real
    
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for n in range(n_band):
        if n == 0:
            ax.plot(k_path.minor_scales, (eig[n] + epr[n]) * Energy.EV["<-"], color="tab:orange", label="w/ EPR")
            ax.plot(k_path.minor_scales, eig[n] * Energy.EV["<-"], color="tab:blue", label="w/o EPR")
        else:
            ax.plot(k_path.minor_scales, (eig[n] + epr[n]) * Energy.EV["<-"], color="tab:orange")
            ax.plot(k_path.minor_scales, eig[n] * Energy.EV["<-"], color="tab:blue")

    for k0 in k_path.major_scales:
        ax.axvline(x=k0, color="black", linewidth=0.3)
    
    ax.set_xticks(k_path.major_scales)
    ax.set_xticklabels(k_names)
    ax.set_ylabel("Energy ($\mathrm{eV}$)")
    ax.set_title("Example: Band structure of bcc-Li")
    ax.set_ylim([-10,20])
    ax.legend()


    fig.savefig("epr.png")

if __name__ == "__main__":
    main()