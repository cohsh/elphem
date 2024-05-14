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
    n_omega = 100

    temperatures = np.arange(0.0, 100.0, 20)

    heat_capacities = np.empty(temperatures.shape)
    heat_capacities_ref = np.empty(temperatures.shape)
    heat_capacities_ref_analytical = np.empty(temperatures.shape)

    i = 0
    for temperature in temperatures:
        lattice = Lattice3D('bcc', 'Li', a)
        electron = FreeElectron.create_from_n(lattice, n_electron, n_band, n_k)
        phonon = DebyePhonon.create_from_n(lattice, debye_temperature, n_q)

        electron_phonon = ElectronPhonon(electron, phonon, sigma=0.0001, eta=0.0001)

        heat_capacities[i] = electron_phonon.calculate_heat_capacity(temperature, n_omega)
        heat_capacities_ref[i] = electron_phonon.calculate_heat_capacity_without_couplings(temperature, n_omega)
        heat_capacities_ref_analytical[i] = electron_phonon.calculate_heat_capacity_without_couplings_analytical(temperature)
        print(i)
        i += 1

    fig = plt.figure()
    ax = {'upper': fig.add_subplot(211), 'lower': fig.add_subplot(212)}
    
    ax['upper'].plot(temperatures, heat_capacities * n_a * Energy.SI['<-'] * 1000.0, color='tab:blue')
    ax['upper'].plot(temperatures, heat_capacities_ref * n_a * Energy.SI['<-'] * 1000.0, color='tab:orange')
    ax['upper'].plot(temperatures, heat_capacities_ref * n_a * Energy.SI['<-'] * 1000.0, color='tab:green')

    ax['lower'].plot(temperatures, (heat_capacities - heat_capacities_ref) * n_a * Energy.SI['<-'] * 1000.0)

    ax['upper'].set_ylabel("Heat Capacity ($\mathrm{mJ}/(\mathrm{K}\cdot \mathrm{mol})$)")
    ax['lower'].set_ylabel("Difference ($\mathrm{mJ}/(\mathrm{K}\cdot \mathrm{mol})$)")

    fig.savefig("heat_capacity.png")

if __name__ == "__main__":
    main()