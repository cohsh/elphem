import numpy as np
import matplotlib.pyplot as plt

from elphem import EmptyLattice, FreeElectron, DebyeModel, SelfEnergy, Spectrum
from elphem.const import Mass, Energy, Length, AtomicWeight, SpecialPoints

def main():
    # Example: Li (BCC)

    a = 2.98 * Length.ANGSTROM["->"]
    mass = AtomicWeight.table["Li"] * Mass.DALTON["->"]

    debye_temperature = 344.0

    lattice = EmptyLattice('bcc', a)
    electron = FreeElectron(lattice, n_band=8, n_electron=1)
    phonon = DebyeModel(lattice, debye_temperature, 1, mass)

    temperature =  1.0 * debye_temperature
    v0 = 1.0 / 16.0
    self_energy = SelfEnergy(lattice, electron, phonon, temperature, sigma=0.0002, eta=0.01, effective_potential=v0)

    n_q = np.array([10]*3)
    n_omega = 1000
    range_omega = [1.0 * Energy.EV["->"], 1.25 * Energy.EV["->"]]
#    range_omega = [-2.0, 2.0]
    
    k_names = ["N", "H"]
    n_split = 100
    
    x, omega, spectrum, special_x = Spectrum(self_energy).calculate_with_path(k_names, n_split, n_q, n_omega, range_omega)

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    ax.plot(omega * Energy.EV["<-"], np.abs(spectrum[0]) / Energy.EV["<-"])
#   ax.axvline(x=0.0, color="black", linewidth=0.3)


    ax.set_xlabel("$\omega$ ($\mathrm{eV}$)")
    ax.set_ylabel("$A(\mathbf{k}_\mathrm{N}, \omega)$ ($\mathrm{eV}^{-1}$)")
    ax.set_title("Spectral function of bcc-Li")

    fig.savefig("test_spectrum_single_k_compare.png")

if __name__ == "__main__":
    main()