"""Example: bcc-Li"""
import time
import numpy as np
import matplotlib.pyplot as plt
from elphem import *

def main():
    a = 2.98 * Length.ANGSTROM['->']
    debye_temperature = 344.0
    n_k = np.full(3, 10)
    n_electron = 1
    n_band = 4

    lattice = Lattice3D('bcc', 'Li', a, debye_temperature)

    electron = FreeElectron.create_from_n(lattice, n_electron, n_band, n_k)

    n_omega = 10000
    range_omega = [-10 * Energy.EV["->"], 10 * Energy.EV["->"]]
    omega_array = np.linspace(range_omega[0] , range_omega[1], n_omega)

    sigma = 0.0001
    
    dos_fermi = electron.calculate_dos(0.0)
    print(dos_fermi)
    
    dos = electron.calculate_dos(omega_array)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(omega_array * Energy.EV["<-"], dos)

    ax.set_xlabel("$\omega$ ($\mathrm{eV}$)")
    ax.set_ylabel("Density of States")

    fig.savefig("dos.png")

if __name__ == "__main__":
    main()