"""Example: bcc-Li"""
import numpy as np
import matplotlib.pyplot as plt
from elphem import *

def main():
    # Parameters of lattice
    a = 2.98 * Length.ANGSTROM['->']

    # Parameters of electron
    n_electrons = 1
    n_bands_electron = 4

    # Parameters of phonon
    debye_temperature = 344.0
    n_q = [6, 6, 6]
    
    # Parameters of k-path
    k_names = ["G", "H", "N", "G", "P", "H"]
    n_split = 50
    
    # Parameters of electron-phonon
    temperature = 0.8 * debye_temperature
    n_bands_elph = 4

    # Generate a lattice
    lattice = Lattice3D('bcc', 'Li', a)

    # Get k-path
    k_path = lattice.get_k_path(k_names, n_split)

    # Generate an electron.
    electron = Electron.create_from_path(lattice, n_electrons, n_bands_electron, k_path)

    # Generate a phonon.
    phonon = Phonon.create_from_n(lattice, debye_temperature, n_q)

    # Generate electron-phonon
    electron_phonon = ElectronPhonon(electron, phonon, temperature, n_bands_elph)
    
    # Calculate electron-phonon renormalization
    epr = electron_phonon.calculate_electron_phonon_renormalization()
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    for i in range(n_bands_electron):
        ax.plot(k_path.minor_scales, electron.eigenenergies[i] * Energy.EV["<-"], color='tab:blue')
        ax.plot(k_path.minor_scales, (electron.eigenenergies[i] + epr[i]) * Energy.EV["<-"], color='tab:orange')
    
    for x0 in k_path.major_scales:
        ax.axvline(x=x0, color="black", linewidth=0.3)
    
    ax.set_xticks(k_path.major_scales)
    ax.set_xticklabels(k_names)
    
    ax.set_ylabel("Energy ($\mathrm{eV}$)")
    ax.set_title("EPR of bcc-Li")

    fig.savefig("epr.png")

if __name__ == "__main__":
    main()