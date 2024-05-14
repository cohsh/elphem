"""Example: bcc-Li"""
import numpy as np
import matplotlib.pyplot as plt
from elphem import *

def main():
    a = 2.98 * Length.ANGSTROM['->']
    debye_temperature = 344.0
    n_q = np.full(3, 20)
    k_names = ["G", "H", "N", "G", "P", "H"]
    n_split = 20
    n_electron = 1
    n_band = 1

    lattice = Lattice3D('bcc', 'Li', a)

    k_path = lattice.reciprocal.get_path(k_names, n_split)

    electron = FreeElectron.create_from_path(lattice, n_electron, n_band, k_path)
    phonon = DebyePhonon.create_from_n(lattice, debye_temperature, n_q)

    electron_phonon = ElectronPhonon(electron, phonon, sigma=0.0001, eta=0.0001)

    n_omega = 100
    range_omega = [-6 * Energy.EV["->"], 5 * Energy.EV["->"]]
    omega_array = np.linspace(range_omega[0] , range_omega[1], n_omega)
    
    spectrum = electron_phonon.calculate_spectrum_over_range(300.0, omega_array)
    
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
    mappable.set_clim(0.0, 0.1)

    fig.savefig("spectrum.png")

if __name__ == "__main__":
    main()