import numpy as np
import matplotlib.pyplot as plt
from elphem import *

def main():
    # Example: Li (BCC)
    a = 2.98 * Length.ANGSTROM["->"]
    debye_temperature = 344.0
    n_band = 1
    n_electron = 1
    k_names = ["N", "H"]
    n_split = 2
    n_q_array = np.full(3, 40)

    lattice = Lattice('bcc', 'Li', a, debye_temperature)
    
    k_path = lattice.reciprocal.get_path(k_names, n_split)
    electron = FreeElectron.create_from_k(lattice, n_electron, n_band, k_path.values)
    phonon = DebyePhonon.create_from_n(lattice, debye_temperature, n_q_array)

    n_omega = 2000
    range_omega = [0.5 * Energy.EV["->"], 2.5 * Energy.EV["->"]]
    omega_array = np.linspace(range_omega[0], range_omega[1], n_omega)
    
    elph = ElectronPhonon(electron, phonon)
    
    spectrum = elph.calculate_spectrum_over_range(omega_array)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(omega_array * Energy.EV["<-"], np.abs(spectrum[0]) / Energy.EV["<-"])
#    ax.axvline(x=0.0, color="black", linewidth=0.3)

    ax.set_ylim(0,30)
    ax.set_xlabel("$\omega$ ($\mathrm{eV}$)")
    ax.set_ylabel("$A(\mathbf{k}_\mathrm{N}, \omega)$ ($\mathrm{eV}^{-1}$)")
    ax.set_title("Spectral function of bcc-Li")

    fig.savefig("spectrum_k.png")

if __name__ == "__main__":
    main()