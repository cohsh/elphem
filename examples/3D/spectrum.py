"""Example: bcc-Li"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from elphem import *

def calculate_3d():
    a = 2.98 * Length.ANGSTROM['->']
    debye_temperature = 344.0
    n_q = np.full(3, 10)
    k_names = ["G", "H", "N", "G", "P", "H"]
    n_split = 40
    n_electron = 1
    n_band = 7

    lattice = Lattice3D('bcc', 'Li', a)

    k_path = lattice.reciprocal.get_path(k_names, n_split)

    electron = FreeElectron.create_from_path(lattice, n_electron, n_band, k_path)
    phonon = DebyePhonon.create_from_n(lattice, debye_temperature, n_q)

    electron_phonon = ElectronPhonon(electron, phonon, sigma=0.001, eta=0.0005)

    n_omega = 100
    range_omega = [-6 * Energy.EV["->"], 20 * Energy.EV["->"]]
    omega_array = np.linspace(range_omega[0] , range_omega[1], n_omega)
    
    spectrum = electron_phonon.calculate_spectrum_over_range(300.0, omega_array)
    
    y, x = np.meshgrid(omega_array, k_path.minor_scales)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    mappable = ax.pcolormesh(x, y * Energy.EV["<-"], spectrum / Energy.EV["<-"])
    
    for x0 in k_path.major_scales:
        ax.axvline(x=x0, color="black", linewidth=0.3)
    
    ax.set_xticks(k_path.major_scales)
    ax.set_xticklabels(k_names)
    ax.set_ylabel("Energy ($\mathrm{eV}$)")
    ax.set_title("Spectral function of bcc-Li")
    
    fig.colorbar(mappable, ax=ax)
    mappable.set_clim(0.0, 0.01)

    fig.savefig("spectrum_3d.png")

def calculate_2d():
    pass

def calculate_1d():
    a = 2.98 * Length.ANGSTROM['->']
    debye_temperature = 344.0
    n_q = np.full(1, 50)
    k_names = ["G", "X"]
    n_split = 100
    n_electron = 1
    n_band = 7

    lattice = Lattice1D('Li', a)

    k_path = lattice.reciprocal.get_path(k_names, n_split)

    electron = FreeElectron.create_from_path(lattice, n_electron, n_band, k_path)
    phonon = DebyePhonon.create_from_n(lattice, debye_temperature, n_q)

    electron_phonon = ElectronPhonon(electron, phonon, sigma=0.001, eta=0.0005)

    n_omega = 100
    range_omega = [-6 * Energy.EV["->"], 20 * Energy.EV["->"]]
    omega_array = np.linspace(range_omega[0] , range_omega[1], n_omega)
    
    spectrum = electron_phonon.calculate_spectrum_over_range(100.0, omega_array)
    
    y, x = np.meshgrid(omega_array, k_path.minor_scales)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    mappable = ax.pcolormesh(x, y * Energy.EV["<-"], spectrum / Energy.EV["<-"])
    
    for x0 in k_path.major_scales:
        ax.axvline(x=x0, color="black", linewidth=0.3)
    
    ax.set_xticks(k_path.major_scales)
    ax.set_xticklabels(k_names)
    ax.set_ylabel("Energy ($\mathrm{eV}$)")
    ax.set_title("Spectral function of bcc-Li")
    
    fig.colorbar(mappable, ax=ax)
    mappable.set_clim(0.0, 0.01)

    fig.savefig("spectrum_1d.png")

if __name__ == "__main__":
    args = sys.argv
    if len(args) < 2:
        raise ValueError('please set n_dim')
    else:
        n_dim = int(args[1])
    
    if n_dim == 3:
        calculate_3d()
    elif n_dim == 2:
        calculate_2d()
    elif n_dim == 1:
        calculate_1d()
    else:
        raise ValueError('n_dim = 1, 2, 3')