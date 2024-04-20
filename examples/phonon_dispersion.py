"""Example: bcc-Li"""
import matplotlib.pyplot as plt
from elphem import *

def main():
    a = 2.98 * Length.ANGSTROM["->"]
    lattice = Lattice('bcc', 'Li', a)

    debye_temperature = 344.0
    phonon = DebyePhonon(lattice, debye_temperature)

    q_names = ["G", "H", "N", "G", "P", "H"]
    
    q, omega, special_q = phonon.get_dispersion(q_names, n_split=40)
    
    fig, ax = plt.subplots()

    ax.plot(q, omega * Energy.EV["<-"] * 1.0e+3, color="tab:blue")
    
    for q0 in special_q:
        ax.axvline(x=q0, color="black", linewidth=0.3)
    
    ax.set_xticks(special_q)
    ax.set_xticklabels(q_names)
    ax.set_ylabel("Energy ($\mathrm{meV}$)")

    fig.savefig("example_phonon_dispersion.png")

if __name__ == "__main__":
    main()