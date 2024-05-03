"""Example: bcc-Li"""
import numpy as np
import matplotlib.pyplot as plt
from elphem import *

def main():
    a = 2.98 * Length.ANGSTROM['->']
    debye_temperature = 344.0
    n_q = np.full(3, 20)
    n_electron = 1
    n_band = 1
    k_gamma = np.full(3, 0.)

    lattice = Lattice('bcc', 'Li', a, debye_temperature)

    electron = FreeElectron.create_from_k(lattice, n_electron, n_band, k_gamma)
    phonon = DebyePhonon.create_from_n(lattice, debye_temperature, n_q)

    print(electron.eigenenergies.shape)

    electron_phonon = ElectronPhonon(electron, phonon)
    
    n_omega = 1000
    range_omega = [-20 * Energy.EV["->"], 20 * Energy.EV["->"]]
    omega_array = np.linspace(range_omega[0] , range_omega[1], n_omega)
    
    self_energies = electron_phonon.calculate_self_energies_over_range(omega_array)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(omega_array * Energy.EV["<-"], self_energies[0, 0] * Energy.EV["<-"])

    ax.set_xlabel("$\omega$ ($\mathrm{eV}$)")
    ax.set_ylabel("$A(\mathbf{k}_\mathrm{N}, \omega)$ ($\mathrm{eV}^{-1}$)")
    ax.set_title("Spectral function of bcc-Li")

    fig.savefig("self_energy.png")

if __name__ == "__main__":
    main()