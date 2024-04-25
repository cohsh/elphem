"""Example: bcc-Li"""
import numpy as np
import matplotlib.pyplot as plt
from elphem import *

def main():
    a = 2.98 * Length.ANGSTROM['->']
    lattice = Lattice('bcc', 'Li', a)

    electron = FreeElectron(lattice, n_band=4, n_electron=1)

    debye_temperature = 344.0
    phonon = DebyePhonon(lattice, debye_temperature)

    n_q = np.full(3, 8)
    temperature = debye_temperature
    electron_phonon = ElectronPhonon(electron, phonon, temperature, n_q, sigma=0.0001)

    n_omega = 100
    range_omega = [-10 * Energy.EV["->"], 20 * Energy.EV["->"]]
#    range_omega = [-2 * Energy.EV["->"], 2 * Energy.EV["->"]]
    
    k_names = ["G", "H", "N", "G", "P", "H"]
    n_split = 20
    
    x, y, spectrum, special_x = SpectrumBW(electron_phonon).get_with_path(k_names, n_split, n_omega, range_omega)
    y_mesh, x_mesh = np.meshgrid(y, x)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    mappable = ax.pcolormesh(x_mesh, y_mesh * Energy.EV["<-"], spectrum / Energy.EV["<-"])
    
    for x0 in special_x:
        ax.axvline(x=x0, color="black", linewidth=0.3)
    
    ax.set_xticks(special_x)
    ax.set_xticklabels(k_names)
    ax.set_ylabel("Energy ($\mathrm{eV}$)")
    ax.set_title("Spectral function of bcc-Li")
    
    fig.colorbar(mappable, ax=ax)
    mappable.set_clim(-3.0, 0.0)

    fig.savefig("spectrum_bw.png")

if __name__ == "__main__":
    main()