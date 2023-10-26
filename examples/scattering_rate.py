import numpy as np
import matplotlib.pyplot as plt

from elphem import LatticeConstant, EmptyLattice, FreeElectron, DebyeModel, SelfEnergy2nd
from elphem.const import Mass, Energy, Prefix, Time, Length, AtomicWeight

def main():
    # Example: Li (BCC)
    a = 2.98 * Length.ANGSTROM["->"]
    alpha = 109.47
    lattice_constant = LatticeConstant(a,a,a,alpha,alpha,alpha)
    lattice = EmptyLattice(lattice_constant)

    electron = FreeElectron(lattice, 1)
    
    mass = AtomicWeight.table["Li"] * Mass.DALTON["->"]
    
    debye_temperature = 344.0

    phonon = DebyeModel(lattice, debye_temperature, 1, mass)

    temperature = debye_temperature

    self_energy = SelfEnergy2nd(lattice, electron, phonon)

    n_g = np.array([1]*3)
    n_k = np.array([6]*3)
    g, k = electron.grid(n_g, n_k)
    
    n_g_inter = np.array([1]*3)
    n_q = np.array([10]*3)
    selfen = self_energy.calculate(temperature, g, k, n_g_inter, n_q)

    epsilon_nk = electron.eigenenergy(k + g)

    fig = plt.figure()

    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)

    for ax in [ax1, ax2]:
        ax.scatter(epsilon_nk * Energy.EV["<-"], selfen.imag / (Time.SI["<-"] / Prefix.PICO), label="$\mathrm{Im}\Sigma^\mathrm{Fan}$")

        ax.set_ylabel("Scattering rate ($\mathrm{ps}^{-1}$)")
        ax.legend()

    ax2.set_xlabel("Electron energy ($\mathrm{eV}$)")
    ax2.set_yscale("log")
    ax1.set_title("Example: Scattering rate of bcc-Li")
    
    file_name = "test_scattering_rate.png"
    fig.savefig(file_name)

if __name__ == "__main__":
    main()