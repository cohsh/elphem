import numpy as np
import matplotlib.pyplot as plt
import random
import string
import os
from unittest import TestCase

from elphem.const.unit import Mass, Energy, Time, Prefix
from elphem.lattice.empty import EmptyLattice, LatticeConstant
from elphem.electron.free import FreeElectron
from elphem.phonon.debye import DebyeModel
from elphem.elph.self_energy import SelfEnergy2nd

class TestUnit(TestCase):
    def test_calculate(self):
        lattice_constant = LatticeConstant(5,5,5,60,60,60)
        lattice = EmptyLattice(lattice_constant)

        electron = FreeElectron(lattice, 4)
        
        mass = 12 * Mass.DALTON["->"]
        
        debye_temperature = 2300.0

        phonon = DebyeModel(lattice, debye_temperature, 2, mass)

        temperature = debye_temperature

        self_energy = SelfEnergy2nd(lattice, electron, phonon)

        n_g = np.array([1]*3)
        n_k = np.array([5]*3)
        g, k = electron.grid(n_g, n_k)
        
        n_g_inter = np.array([1]*3)
        n_q = np.array([5]*3)
        selfen = self_energy.calculate(temperature, g, k, n_g_inter, n_q)
        
        self.assertEqual(selfen.shape, (n_g[0]*2, n_g[1]*2, n_g[2]*2) + (n_k[0], n_k[1], n_k[2]))
    
    def test_save_imag(self):
        lattice_constant = LatticeConstant(5,5,5,60,60,60)
        lattice = EmptyLattice(lattice_constant)

        electron = FreeElectron(lattice, 4)
        
        mass = 12 * Mass.DALTON["->"]
        
        debye_temperature = 2300.0

        phonon = DebyeModel(lattice, debye_temperature, 2, mass)

        temperature = debye_temperature

        self_energy = SelfEnergy2nd(lattice, electron, phonon)

        n_g = np.array([1]*3)
        n_k = np.array([5]*3)
        g, k = electron.grid(n_g, n_k)
        
        n_g_inter = np.array([1]*3)
        n_q = np.array([5]*3)
        selfen = self_energy.calculate(temperature, g, k, n_g_inter, n_q)

        epsilon_nk = electron.eigenenergy(k + g)

        fig = plt.figure()

        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2)

        for ax in [ax1, ax2]:
            ax.scatter(epsilon_nk * Energy.EV["<-"], selfen.imag / (Time.SI["<-"] / Prefix.PICO), label="$\Sigma^\mathrm{2nd}$")

            ax.set_ylabel("Scattering rate ($\mathrm{ps}^{-1}$)")

            ax.legend()

        ax2.set_xlabel("Electron energy ($\mathrm{eV}$)")
        ax2.set_yscale("log")
        ax2.set_ylim([1.0e-5 / (Time.SI["<-"] / Prefix.PICO), selfen.imag.max() / (Time.SI["<-"] / Prefix.PICO) * 10])
        
        file_name = "".join(random.choices(string.ascii_letters + string.digits, k=20)) + ".png"
        fig.savefig(file_name)
        
        is_file = os.path.isfile(file_name)
        if is_file:
            os.remove(file_name)
        
        self.assertTrue(is_file)