"""Example: bcc-Li"""
import time
import numpy as np
import matplotlib.pyplot as plt
from elphem import *

def main():
    temperature = 344.0
    kbt = temperature * Energy.KELVIN['->']
    print("{} eV".format(kbt * Energy.EV['<-']))

    n_omega = 10000
    range_omega = [-1 * Energy.EV["->"], 1 * Energy.EV["->"]]
    omega_array = np.linspace(range_omega[0] , range_omega[1], n_omega)
    
    fig = plt.figure()
    ax = {'fermi': fig.add_subplot(211), 'bose': fig.add_subplot(212)}


    for temperature in np.arange(0, 1000, 200):
        fermi = fermi_distribution(temperature, omega_array)
        bose = bose_distribution(temperature, omega_array)

        ax['fermi'].plot(omega_array * Energy.EV["<-"], fermi, label="{} K".format(int(temperature)))
        ax['bose'].plot(omega_array * Energy.EV["<-"], bose, label="{} K".format(int(temperature)))

    ax['bose'].set_xlim([0, 0.2])
    ax['bose'].set_ylim([0, 1])

    ax['bose'].set_xlabel("$\omega$ ($\mathrm{eV}$)")
    ax['fermi'].set_ylabel("Fermi Distribution")
    ax['bose'].set_ylabel("Bose Distribution")

    ax['fermi'].set_title("Fermi and Bose distributions")

    ax['fermi'].legend()
    ax['bose'].legend()

    fig.savefig("distributions.png")

if __name__ == "__main__":
    main()