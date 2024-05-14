"""Example: bcc-Li"""
import numpy as np
import matplotlib.pyplot as plt
from elphem import *

n_a = 6.02214085774 * 10 ** 23

def main():
    a = 2.98 * Length.ANGSTROM['->']
    debye_temperature = 344.0
    n_k = np.full(3, 4)
    n_q = np.full(3, 8)
    n_electron = 1
    n_band = 1

    temperatures = np.arange(0.0, 300.0, 5)

    heat_capacities = np.empty(temperatures.shape)

    i = 0
    for temperature in temperatures:
        lattice = Lattice3D('bcc', 'Li', a, temperature)
        electron = FreeElectron.create_from_n(lattice, n_electron, n_band, n_k)
        phonon = DebyePhonon.create_from_n(lattice, debye_temperature, n_q)

        electron_phonon = ElectronPhonon(electron, phonon, sigma=0.0001, eta=0.0001)

        heat_capacities[i] = electron_phonon.calculate_heat_capacity()
        print(i)
        i += 1

    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    ax.plot(temperatures, heat_capacities * n_a * Energy.SI['<-'])
    
    ax.set_ylabel("Heat Capacity ($\mathrm{J}/(\mathrm{K}\cdot \mathrm{mol})$)")

    fig.savefig("heat_capacity.png")

if __name__ == "__main__":
    main()