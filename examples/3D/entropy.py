"""Example: bcc-Li"""
import numpy as np
import matplotlib.pyplot as plt
from elphem import *

def main():
    a = 2.98 * Length.ANGSTROM['->']
    debye_temperature = 344.0
    n_k = np.full(3, 4)
    n_q = np.full(3, 20)
    n_electron = 1
    n_band = 1

    lattice = Lattice3D('bcc', 'Li', a, debye_temperature)

    electron = FreeElectron.create_from_n(lattice, n_electron, n_band, n_k)
    phonon = DebyePhonon.create_from_n(lattice, debye_temperature, n_q)

    electron_phonon = ElectronPhonon(electron, phonon, sigma=0.001, eta=0.0005)

    n_omega = 100
    
    entropy = electron_phonon.calculate_entropy(n_omega)
    
    print(entropy)

if __name__ == "__main__":
    main()