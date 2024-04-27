"""Example: bcc-Li"""
import numpy as np
import matplotlib.pyplot as plt
from elphem import *

def main():
    a = 2.98 * Length.ANGSTROM['->']
    lattice = Lattice('bcc', 'Li', a)

    electron = FreeElectron(lattice, n_band=1, n_electron=1)

    debye_temperature = 344.0
    phonon = DebyePhonon(lattice, debye_temperature)

    n_q = np.full(3, 20)
    temperature = debye_temperature
    electron_phonon = ElectronPhonon(electron, phonon, temperature, n_q)

    n_omega = 50
    range_omega = [-6 * Energy.EV["->"], 10 * Energy.EV["->"]]
    
    k_names = ["G", "H", "N", "G", "P", "H"]
    n_split = 20
    
    k, omega, spectrum, special_k = electron_phonon.get_spectrum(k_names, n_split, n_omega, range_omega)
    omega_mesh, k_mesh = np.meshgrid(omega, k)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    mappable = ax.pcolormesh(k_mesh, omega_mesh * Energy.EV["<-"], spectrum / Energy.EV["<-"])
    
    for x0 in special_k:
        ax.axvline(x=x0, color="black", linewidth=0.3)
    
    ax.set_xticks(special_k)
    ax.set_xticklabels(k_names)
    ax.set_ylabel("Energy ($\mathrm{eV}$)")
    ax.set_title("Spectral function of bcc-Li")
    
    fig.colorbar(mappable, ax=ax)
    mappable.set_clim(-1.0, 0.0)

    fig.savefig("spectrum.png")

if __name__ == "__main__":
    main()