"""Example: bcc-Li"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from elphem import *

def main():
    a = 2.98 * Length.ANGSTROM['->']
    debye_temperature = 344.0
    n_q = [10, 10, 10]
    k_names = ["G", "H", "N", "G", "P", "H"]
    n_split = 40
    n_electron = 1
    n_band = 7

    lattice = Lattice3D('bcc', 'Li', a)

    k_path = lattice.get_k_path(k_names, n_split)

    # Generate an electron.
    electron = Electron.create_from_path(lattice, n_electron, n_band, k_path)

    # Generate a phonon.
    phonon = Phonon.create_from_n(lattice, debye_temperature, n_q)

    electron_phonon = ElectronPhonon(electron, phonon, debye_temperature, sigma=0.001, eta=0.0005)

    n_omega = 100
    range_omega = [-6 * Energy.EV["->"], 20 * Energy.EV["->"]]
    omega_array = np.linspace(range_omega[0] , range_omega[1], n_omega)
    
    spectrum = electron_phonon.calculate_spectrum_over_range(omega_array, normalize=True)
    
    y, x = np.meshgrid(omega_array, k_path.minor_scales)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    mappable = ax.pcolormesh(x, y * Energy.EV["<-"], spectrum / Energy.EV["<-"])
    
    for x0 in k_path.major_scales:
        ax.axvline(x=x0, color="black", linewidth=0.3)
    
    ax.set_xticks(k_path.major_scales)
    ax.set_xticklabels(k_names)
    ax.set_ylabel("Energy ($\mathrm{eV}$)")
    ax.set_title("Spectral function of bcc-Li")
    
    fig.colorbar(mappable, ax=ax)
    mappable.set_clim(0.01, 0.05)

    fig.savefig("spectrum.png")

if __name__ == "__main__":
    main()