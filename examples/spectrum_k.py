import numpy as np
import matplotlib.pyplot as plt
from elphem import *

def main():
    # Example: Li (BCC)
    a = 2.98 * Length.ANGSTROM["->"]
    debye_temperature = 344.0

    lattice = Lattice('bcc', 'Li', a)
    electron = FreeElectron(lattice, n_band=2, n_electron=1)
    phonon = DebyePhonon(lattice, debye_temperature)

    n_q = np.full(3, 25)
    temperature = debye_temperature

    n_omega = 1000
    range_omega = [1. * Energy.EV["->"], 1.8 * Energy.EV["->"]]
    
    elph = ElectronPhonon(electron, phonon, temperature, n_q, sigma=0.00001)
    
    k_names = ["N", "H"]
    n_split = 2
    
    _, omega, spectrum, __ = elph.get_spectrum(k_names, n_split, n_omega, range_omega)
 
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(omega * Energy.EV["<-"], np.abs(spectrum[0]) / Energy.EV["<-"])
#    ax.axvline(x=0.0, color="black", linewidth=0.3)

    ax.set_ylim(0,10)
    ax.set_xlabel("$\omega$ ($\mathrm{eV}$)")
    ax.set_ylabel("$A(\mathbf{k}_\mathrm{N}, \omega)$ ($\mathrm{eV}^{-1}$)")
    ax.set_title("Spectral function of bcc-Li")

    fig.savefig("spectrum_k.png")

if __name__ == "__main__":
    main()