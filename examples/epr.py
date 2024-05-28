"""Example: bcc-Li"""
import numpy as np
import matplotlib.pyplot as plt
from elphem import *

def main():
    # Parameters of lattice
    a = 2.98 * Length.ANGSTROM['->']

    # Parameters of electron
    n_electrons = 1
    n_bands_electron = 1

    # Parameters of phonon
    debye_temperature = 344.0
    n_q = [8, 8, 8]
    
    # Parameters of k-path
    k_names = ["G", "H", "N", "G", "P", "H"]
    n_split = 20
    
    # Parameters of electron-phonon
    temperature = 300.0
    n_bands_elph = 1

    # Generate a lattice
    lattice = Lattice3D('bcc', 'Li', a)

    # Get k-path
    k_path = lattice.get_k_path(k_names, n_split)

    # Generate an electron.
    electron = Electron.create_from_path(lattice, n_electrons, n_bands_electron, k_path)

    # Generate a phonon.
    phonon = Phonon.create_from_n(lattice, debye_temperature, n_q)

    # Generate electron-phonon
    electron_phonon = ElectronPhonon(electron, phonon, temperature, n_bands_elph, eta=0.05, coupling_type="bardeen")
    
    # Calculate electron-phonon renormalization
    epr = electron_phonon.calculate_electron_phonon_renormalization()
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    for i in range(n_bands_elph):
        ax.plot(k_path.minor_scales, electron.eigenenergies[i] * Energy.EV["<-"], color='tab:blue', label='w/o EPR')
        ax.plot(k_path.minor_scales, (electron.eigenenergies[i] + epr[i]) * Energy.EV["<-"], color='tab:orange', label='w/ EPR')
    
    for x0 in k_path.major_scales:
        ax.axvline(x=x0, color="black", linewidth=0.3)

    ax.legend()
    
    ax.set_xticks(k_path.major_scales)
    ax.set_xticklabels(k_names)
    
    ax.set_ylabel("Energy ($\mathrm{eV}$)")
    ax.set_title("EPR of bcc-Li ($T=300~\mathrm{K}$)")

    fig.savefig("epr.png")

if __name__ == "__main__":
    main()