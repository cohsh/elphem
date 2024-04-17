import numpy as np
import matplotlib.pyplot as plt
from elphem import *

def main():
    # Example: Li (BCC)
    a = 2.98 * Length.ANGSTROM["->"]
    mass = AtomicWeight.table["Li"] * Mass.DALTON["->"]
    debye_temperature = 344.0
    temperature = 0.3 * debye_temperature
    n_band = 20

    lattice = EmptyLattice('bcc', a)
    electron = FreeElectron(lattice, n_band, 1)        
    phonon = DebyeModel(lattice, temperature, 1, mass)

    self_energy = SelfEnergy(lattice, electron, phonon, temperature, eta=0.1)

    k_names = ["G", "H", "N", "G", "P", "H"]

    n_split = 20
    n_q = np.array([8]*3)
    
    x, eig, epr, special_x = EPR(self_energy).calculate_with_path(k_names, n_split, n_q)
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    for n in range(n_band):
        ax.plot(x, eig[n] * Energy.EV["<-"], color="tab:blue")
        ax.plot(x, (eig[n] + epr[n]) * Energy.EV["<-"], color="tab:orange")
    
    for x0 in special_x:
        ax.axvline(x=x0, color="black", linewidth=0.3)
    
    ax.set_xticks(special_x)
    ax.set_xticklabels(k_names)
    ax.set_ylabel("Energy ($\mathrm{eV}$)")
    ax.set_title("Example: Band structure of bcc-Li")

    fig.savefig("test_epr.png")

if __name__ == "__main__":
    main()