"""Example: bcc-Li"""
import matplotlib.pyplot as plt
from elphem import *

def main():
    a = 2.98 * Length.ANGSTROM["->"]
    lattice = Lattice('bcc', 'Li', a)

    debye_temperature = 344.0
    phonon = DebyePhonon(lattice, debye_temperature)

    q_names = ["G", "H", "N", "G", "P", "H"]
    
    omega = phonon.get_dispersion(q_names, n_split=40)
    
    fig, ax = plt.subplots()

    ax.plot(omega.distances, omega.values * Energy.EV["<-"] * 1.0e+3, color="tab:blue")
    
    for q0 in omega.special_distances:
        ax.axvline(x=q0, color="black", linewidth=0.3)
    
    ax.set_xticks(omega.special_distances)
    ax.set_xticklabels(q_names)
    ax.set_ylabel("Energy ($\mathrm{meV}$)")

    fig.savefig("phonon_dispersion.png")

if __name__ == "__main__":
    main()