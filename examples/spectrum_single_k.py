import numpy as np
import matplotlib.pyplot as plt
from elphem import *

def main():
    # Example: Li (BCC)
    a = 2.98 * Length.ANGSTROM["->"]
    debye_temperature = 344.0

    lattice = Lattice('bcc', 'Li', a)
    electron = FreeElectron(lattice, n_band=8, n_electron=1)
    phonon = DebyePhonon(lattice, debye_temperature)

    temperature = debye_temperature
    elph = ElectronPhonon(electron, phonon, temperature, sigma=0.0001)

    n_q = np.full(3, 30)
    n_omega = 1000
    range_omega = [-4.0 * Energy.EV["->"], 4.0 * Energy.EV["->"]]
    
    k_names = ["N", "H"]
    n_split = 100
    
    x, omega, spectrum, special_x = Spectrum(elph).get_with_path(k_names, n_split, n_q, n_omega, range_omega)
 
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(omega * Energy.EV["<-"], np.abs(spectrum[0]) / Energy.EV["<-"])
    ax.axvline(x=0.0, color="black", linewidth=0.3)


    ax.set_xlabel("$\omega$ ($\mathrm{eV}$)")
    ax.set_ylabel("$A(\mathbf{k}_\mathrm{N}, \omega)$ ($\mathrm{eV}^{-1}$)")
    ax.set_title("Spectral function of bcc-Li")

    fig.savefig("example_spectrum_single_k.png")

if __name__ == "__main__":
    main()