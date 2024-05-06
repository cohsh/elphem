"""Example: bcc-Li"""
import time
import numpy as np
import matplotlib.pyplot as plt
from elphem import *

def main():
    a = 2.98 * Length.ANGSTROM['->']
    debye_temperature = 344.0
    n_q = np.full(3, 30)
    n_electron = 1
    n_band = 1

    lattice = Lattice3D('bcc', 'Li', a, debye_temperature)
    
    k = lattice.reciprocal.calculate_special_k('N')

    electron = FreeElectron.create_from_k(lattice, n_electron, n_band, k)
    phonon = DebyePhonon.create_from_n(lattice, debye_temperature, n_q)

    n_omega = 10000
    range_omega = [-10 * Energy.EV["->"], 10 * Energy.EV["->"]]
    omega_array = np.linspace(range_omega[0] , range_omega[1], n_omega)

    sigma = (range_omega[1] - range_omega[0]) / n_omega * 0.1
    
    print("sigma: {} meV".format(sigma * Energy.EV["->"] * 1e+3))
    time.sleep(3)

    electron_phonon = ElectronPhonon(electron, phonon, sigma=sigma, eta=sigma)
    
    
    self_energies = electron_phonon.calculate_self_energies_over_range(omega_array)
    
    fig = plt.figure()
    ax = {'real': fig.add_subplot(211), 'imag': fig.add_subplot(212)}

    ax['real'].plot(omega_array * Energy.EV["<-"], self_energies[0, 0].real * Energy.EV["<-"] * 1000.0)
    ax['imag'].plot(omega_array * Energy.EV["<-"], self_energies[0, 0].imag * Energy.EV["<-"] * 1000.0)

    ax['imag'].set_xlabel("$\omega$ ($\mathrm{eV}$)")
    ax['real'].set_ylabel("$\mathrm{Re}\Sigma_{n=0}(\mathbf{k}_\Gamma, \omega)$ ($\mathrm{meV}$)")
    ax['imag'].set_ylabel("$\mathrm{Im}\Sigma_{n=0}(\mathbf{k}_\Gamma, \omega)$ ($\mathrm{meV}$)")

    ax['real'].set_title("Self-energy of bcc-Li")

    fig.savefig("self_energy.png")

if __name__ == "__main__":
    main()