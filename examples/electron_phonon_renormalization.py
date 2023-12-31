import numpy as np
import matplotlib.pyplot as plt

from elphem import LatticeConstant, EmptyLattice, FreeElectron, DebyeModel, SelfEnergy2nd
from elphem.const import Mass, Energy, Length, AtomicWeight, SpecialPoints

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

    self_energy = SelfEnergy2nd(lattice, electron, phonon)

    n_g = np.array([2]*3)
    
    g = lattice.grid(n_g).reshape(-1, 3)
    
    n_g_inter = np.array([1]*3)
    n_q = np.array([10]*3)

    k_names = ["G", "H", "N", "G", "P", "H"]

    k_via = [SpecialPoints.BCC[name] for name in k_names]

    x, k, special_x = lattice.reciprocal_cell.path(20, *k_via)
    
    selfen = np.empty((len(k)), dtype=complex)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    temperature = 2 * debye_temperature

    for n in range(len(g)):
        eig = electron.eigenenergy(k + g[n])
        for i in range(len(k)):
            selfen[i] = self_energy.calculate(temperature, g[n], k[i], n_g_inter, n_q)

        epr = selfen.real

        ax.plot(x, eig * Energy.EV["<-"], color="tab:blue")
        ax.plot(x, (eig + epr) * Energy.EV["<-"], color="tab:orange")
    
    for x0 in special_x:
        ax.axvline(x=x0, color="black", linewidth=0.3)
    
    ax.set_xticks(special_x)
    ax.set_xticklabels(k_names)
    ax.set_ylabel("Energy ($\mathrm{eV}$)")
    ax.set_ylim([-7,Energy.EV["<-"]])
    ax.set_title("Example: Band structure of bcc-Li")

    fig.savefig("test_epr.png")

if __name__ == "__main__":
    main()