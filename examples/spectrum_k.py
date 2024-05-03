import numpy as np
import matplotlib.pyplot as plt
from elphem import *

def main():
    # Example: Li (BCC)
    a = 2.98 * Length.ANGSTROM["->"]
    debye_temperature = 344.0
    n_electron = 1
    n_band = 2
    n_q_array = np.full(3, 30)

    lattice = Lattice('bcc', 'Li', a, debye_temperature)

    k = lattice.reciprocal.calculate_special_k('N')
    
    electron = FreeElectron.create_from_k(lattice, n_electron, n_band, k)
    phonon = DebyePhonon.create_from_n(lattice, debye_temperature, n_q_array)

    n_omega = 10000
    range_omega = [1.0 * Energy.EV["->"], 1.5 * Energy.EV["->"]]
    omega_array = np.linspace(range_omega[0], range_omega[1], n_omega)
    
    electron_phonon = ElectronPhonon(electron, phonon)
    
    spectrum = electron_phonon.calculate_spectrum_over_range(omega_array)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(omega_array * Energy.EV["<-"], np.abs(spectrum[0]) / Energy.EV["<-"])

    ax.set_ylim(0,50)
    ax.set_xlabel("$\omega$ ($\mathrm{eV}$)")
    ax.set_ylabel("$A(\mathbf{k}_\mathrm{N}, \omega)$ ($\mathrm{eV}^{-1}$)")
    ax.set_title("Spectral function of bcc-Li")

    fig.savefig("spectrum_k.png")

if __name__ == "__main__":
    main()