"""Example: bcc-Li"""
import matplotlib.pyplot as plt
from elphem import *

def main():
    a = 2.98 * Length.ANGSTROM["->"]
    lattice = Lattice3D('bcc', 'Li', a)

    q_names = ["G", "H", "N", "G", "P", "H"]
    q_path = lattice.reciprocal.get_path(q_names, 40)

    debye_temperature = 344.0
    phonon = Phonon.create_from_path(lattice, debye_temperature, q_path)
    
    omega = phonon.eigenenergies
    
    fig, ax = plt.subplots()

    ax.plot(q_path.minor_scales, omega * Energy.EV["<-"] * 1.0e+3, color="tab:blue")
    
    for q0 in q_path.major_scales:
        ax.axvline(x=q0, color="black", linewidth=0.3)
    
    ax.set_xticks(q_path.major_scales)
    ax.set_xticklabels(q_names)
    ax.set_ylabel("Energy ($\mathrm{meV}$)")

    fig.savefig("phonon_dispersion.png")

if __name__ == "__main__":
    main()