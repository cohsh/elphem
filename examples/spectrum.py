import numpy as np
import matplotlib.pyplot as plt

from elphem import EmptyLattice, FreeElectron, DebyeModel, SelfEnergy, Spectrum
from elphem.const import Mass, Energy, Length, AtomicWeight, SpecialPoints

def main():
    # Example: Li (BCC)

    a = 2.98 * Length.ANGSTROM["->"]
    alpha = 109.47
    mass = AtomicWeight.table["Li"] * Mass.DALTON["->"]
    debye_temperature = 344.0
    temperature = 1000.0
    n_band = 10

    lattice = EmptyLattice(a,a,a,alpha,alpha,alpha)
    electron = FreeElectron(lattice, n_band, 1)
    phonon = DebyeModel(lattice, debye_temperature, 1, mass)

    temperature = 2 * debye_temperature
    self_energy = SelfEnergy(lattice, electron, phonon, temperature, sigma=0.01, eta=0.01)

    n_q = np.array([10]*3)
    n_omega = 100
    range_omega = [-0.5, 1.5]
    
    k_names = ["G", "H", "N", "G", "P", "H"]
    k_via = [SpecialPoints.BCC[name] for name in k_names]
    n_via = 50
    
    x, y, spectrum, special_x = Spectrum(self_energy).calculate_with_path(k_via, n_via, n_q, n_omega, range_omega)
    y_mesh, x_mesh = np.meshgrid(y, x)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    
    mappable = ax.pcolormesh(x_mesh, y_mesh * Energy.EV["<-"], spectrum / Energy.EV["<-"])
    
    for x0 in special_x:
        ax.axvline(x=x0, color="black", linewidth=0.3)
    
    ax.set_xticks(special_x)
    ax.set_xticklabels(k_names)
    ax.set_ylabel("Energy ($\mathrm{eV}$)")
    ax.set_title("Spectral function of bcc-Li")
    
    fig.colorbar(mappable, ax=ax)
#    mappable.set_clim(-7.0, 0.0)

    fig.savefig("test_spectrum.png")

if __name__ == "__main__":
    main()