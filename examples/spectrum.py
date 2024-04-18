"""Example: bcc-Li"""
import numpy as np
import matplotlib.pyplot as plt
from elphem import *

def main():
    a = 2.98 * Length.ANGSTROM["->"]
    lattice = Lattice('bcc', a)

    electron = FreeElectron(lattice, n_band=8, n_electron=1)

    debye_temperature = 344.0
    mass = AtomicWeight.table["Li"] * Mass.DALTON["->"]
    phonon = DebyePhonon(lattice, debye_temperature, 1, mass)

    temperature =  1.5 * debye_temperature
    self_energy = SelfEnergy(lattice, electron, phonon, temperature)

    n_q = np.full(3, 12)
    n_omega = 1000
    range_omega = [-10 * Energy.EV["->"], 20 * Energy.EV["->"]]
    
    k_names = ["G", "H", "N", "G", "P", "H"]
    n_split = 20
    
    x, y, spectrum, special_x = Spectrum(self_energy).calculate_with_path(k_names, n_split, n_q, n_omega, range_omega)
    y_mesh, x_mesh = np.meshgrid(y, x)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    
    mappable = ax.pcolormesh(x_mesh, y_mesh * Energy.EV["<-"], spectrum / Energy.EV["<-"])
    
    for x0 in special_x:
        ax.axvline(x=x0, color="black", linewidth=0.3)
    
    ax.set_xticks(special_x)
    ax.set_xticklabels(k_names)
    ax.set_ylabel("Energy ($\mathrm{eV}$)")
    ax.set_title("Spectral function of bcc-Li")
    
    fig.colorbar(mappable, ax=ax)
    mappable.set_clim(-3.0, 0.0)

    fig.savefig("example_spectrum.png")

if __name__ == "__main__":
    main()