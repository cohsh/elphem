"""Example: bcc-Li"""
import time
import numpy as np
import matplotlib.pyplot as plt
from elphem import *

def main():
    a = 2.98 * Length.ANGSTROM['->']
    debye_temperature = 344.0
    n_q = np.full(1, 100)
    n_electron = 1
    n_band = 2

    lattice = Lattice1D('Li', a, debye_temperature)
    
    k_G = lattice.reciprocal.calculate_special_k('G')
    k_X = lattice.reciprocal.calculate_special_k('X')
    
    k = (k_X - k_G) * 0.0

    electron = FreeElectron.create_from_k(lattice, n_electron, n_band, k)
    phonon = DebyePhonon.create_from_n(lattice, debye_temperature, n_q)

    n_omega = 1000
    range_omega = [-2 * Energy.EV["->"], 25 * Energy.EV["->"]]
    omega_array = np.linspace(range_omega[0] , range_omega[1], n_omega)

    electron_phonon = ElectronPhonon(electron, phonon, sigma=0.001, eta=1.0)
    
    
    self_energies = electron_phonon.calculate_self_energies_over_range(omega_array)

    fig = plt.figure()
    ax = {'real': fig.add_subplot(211), 'imag': fig.add_subplot(212)}


    for i in range(n_band):
        ax['real'].plot(omega_array * Energy.EV["<-"], self_energies[i, 0].real * Energy.EV["<-"] * 1000.0, label='$n={}$'.format(i), alpha=0.3)
        ax['imag'].plot(omega_array * Energy.EV["<-"], self_energies[i, 0].imag * Energy.EV["<-"] * 1000.0, label='$n={}$'.format(i), alpha=0.3)

    ax['real'].legend()
    ax['imag'].legend()

    ax['imag'].set_xlabel("$\omega$ ($\mathrm{eV}$)")
    ax['real'].set_ylabel("$\mathrm{Re}\Sigma_{n=0}(\mathbf{k}_\Gamma, \omega)$ ($\mathrm{meV}$)")
    ax['imag'].set_ylabel("$\mathrm{Im}\Sigma_{n=0}(\mathbf{k}_\Gamma, \omega)$ ($\mathrm{meV}$)")

    ax['real'].set_title("Self-energy of bcc-Li")

    fig.savefig("self_energy.png")

if __name__ == "__main__":
    main()