"""Example: bcc-Li"""
import numpy as np
import matplotlib.pyplot as plt
from elphem import *

def main():
    a = 2.98 * Length.ANGSTROM['->']
    debye_temperature = 344.0
    n_k = np.full(3, 4)
    n_q = np.full(3, 8)
    n_electron = 1
    n_band = 1

    lattice = Lattice3D('bcc', 'Li', a, 100.0)

    electron = FreeElectron.create_from_n(lattice, n_electron, n_band, n_k)

    phonon = DebyePhonon.create_from_n(lattice, debye_temperature, n_q)

    electron_phonon = ElectronPhonon(electron, phonon, sigma=0.0001, eta=0.0001)

    print(electron_phonon.electron.n_band)

    heat_capacity = electron_phonon.calculate_heat_capacity()

    print(heat_capacity)

if __name__ == "__main__":
    main()