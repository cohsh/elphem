# Elphem
[![Upload Python Package](https://github.com/cohsh/elphem/actions/workflows/python-publish.yml/badge.svg)](https://github.com/cohsh/elphem/actions/workflows/python-publish.yml)
[![Python package](https://github.com/cohsh/elphem/actions/workflows/python-package.yml/badge.svg)](https://github.com/cohsh/elphem/actions/workflows/python-package.yml)
![Python Version](https://img.shields.io/badge/Python-3.10%2C%203.11-blue?logo=python)
![GitHub](https://img.shields.io/github/license/cohsh/elphem)

Python Library for Calculations of **El**ectron-**Ph**onon Interactions with **Em**pty Lattice

## Support
- Python 3.10 and 3.11

## Installation
```shell
git clone git@github.com:cohsh/elphem.git
cd elphem
pip install .
```

## Features
Currently, Elphem allows calculations of
- (reciprocal) lattice vectors from lattice constants.
- electronic structures with empty lattice approximation.
- phonon dispersion relations with Debye model.
- first-order electron-phonon couplings.
- one-electron self-energies.

## Examples
### Calculation of the electronic band structure

![band structure](images/band_structure.png)

```python
import numpy as np
import matplotlib.pyplot as plt

from elphem import SpecialPoints, Energy, Length, LatticeConstant, EmptyLattice, FreeElectron

def main():
    # Example: Li (BCC)
    a = 2.98 * Length.ANGSTROM["->"]
    alpha = 109.47

    lattice_constant = LatticeConstant(a,a,a,alpha,alpha,alpha)
    lattice = EmptyLattice(lattice_constant)
    
    n_cut = np.array([2] * 3)
    electron = FreeElectron(lattice, 1)
        
    k_names = ["G", "H", "N", "G", "P", "H"]
    k_via = [SpecialPoints.BCC[name] for name in k_names]

    x, eig, x_special = electron.get_band_structure(n_cut, *k_via)

    fig, ax = plt.subplots()
    for band in eig:
        ax.plot(x, band * Energy.EV["<-"], color="tab:blue")
    
    ax.vlines(x_special, ymin=-10, ymax=50, color="black", linewidth=0.3)
    ax.set_xticks(x_special)
    ax.set_xticklabels(k_names)
    ax.set_ylabel("Energy ($\mathrm{eV}$)")
    ax.set_ylim([-10,50])

    fig.savefig("test_band_structure.png")

if __name__ == "__main__":
    main()
```

### Calculation of the phonon dispersion

![phonon dispersion](images/phonon_dispersion.png)

```python
import os
import matplotlib.pyplot as plt

from elphem import LatticeConstant, EmptyLattice, DebyeModel, AtomicWeight, Energy, Mass, Length, SpecialPoints, Prefix

def main():
    # Example: \gamma-Fe (FCC)
    a = 2.58 * Length.ANGSTROM["->"]
    lattice_constant = LatticeConstant(a, a, a, 60, 60, 60)
    lattice = EmptyLattice(lattice_constant)
    
    debye_temperature = 470.0

    phonon = DebyeModel(lattice, debye_temperature, 1, AtomicWeight.table["Fe"] * Mass.DALTON["->"])

    q_names = ["G", "X", "G", "L"]
    q_via = [SpecialPoints.FCC[name] for name in q_names]
    
    x, omega, x_special = phonon.get_dispersion(*q_via)
    
    fig, ax = plt.subplots()

    ax.plot(x, omega * Energy.EV["<-"] / Prefix.MILLI, color="tab:blue")
    
    for x0 in x_special:
        ax.axvline(x=x0, color="black", linewidth=0.3)
    
    ax.set_xticks(x_special)
    ax.set_xticklabels(q_names)
    ax.set_ylabel("Energy ($\mathrm{meV}$)")

    fig.savefig("test_phonon_dispersion.png")

if __name__ == "__main__":
    main()
```

### Calculation of the electron-phonon renormalization (EPR)

![epr](images/epr.png)

```python
import numpy as np
import matplotlib.pyplot as plt

from elphem import LatticeConstant, EmptyLattice, FreeElectron, DebyeModel, SelfEnergy2nd
from elphem.const import Mass, Energy, Length, AtomicWeight, SpecialPoints

def main():
    # Example: Li (BCC)
    a = 2.98 * Length.ANGSTROM["->"]
    alpha = 109.47
    lattice_constant = LatticeConstant(a,a,a,alpha,alpha,alpha)
    lattice = EmptyLattice(lattice_constant)

    electron = FreeElectron(lattice, 1)
    
    mass = AtomicWeight.table["Li"] * Mass.DALTON["->"]
    
    debye_temperature = 344.0

    phonon = DebyeModel(lattice, debye_temperature, 1, mass)

    self_energy = SelfEnergy2nd(lattice, electron, phonon)

    n_g = np.array([2]*3)
    
    g = lattice.grid(n_g).reshape(-1, 3)
    
    n_g_inter = np.array([1]*3)
    n_q = np.array([10]*3)

    k_names = ["G", "H", "N", "G", "P", "H"]

    k_via = [SpecialPoints.BCC[name] for name in k_names]

    x, k, special_x = lattice.reciprocal_cell.path(20, *k_via)
    
    selfen = np.empty((len(k)), dtype=complex)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    temperature = 2 * debye_temperature

    for n in range(len(g)):
        eig = electron.eigenenergy(k + g[n])
        for i in range(len(k)):
            selfen[i] = self_energy.calculate(temperature, g[n], k[i], n_g_inter, n_q)

        epr = selfen.real

        ax.plot(x, eig * Energy.EV["<-"], color="tab:blue")
        ax.plot(x, (eig + epr) * Energy.EV["<-"], color="tab:orange")
    
    for x0 in special_x:
        ax.axvline(x=x0, color="black", linewidth=0.3)
    
    ax.set_xticks(special_x)
    ax.set_xticklabels(k_names)
    ax.set_ylabel("Energy ($\mathrm{eV}$)")
    ax.set_ylim([-7,Energy.EV["<-"]])
    ax.set_title("Example: Band structure of bcc-Li")

    fig.savefig("test_epr.png")

if __name__ == "__main__":
    main()
```

### Calculation of scattering rate

![scattering_rate](images/scattering_rate.png)

```python
import numpy as np
import matplotlib.pyplot as plt

from elphem import LatticeConstant, EmptyLattice, FreeElectron, DebyeModel, SelfEnergy2nd
from elphem.const import Mass, Energy, Prefix, Time, Length, AtomicWeight

def main():
    # Example: Li (BCC)
    a = 2.98 * Length.ANGSTROM["->"]
    alpha = 109.47
    lattice_constant = LatticeConstant(a,a,a,alpha,alpha,alpha)
    lattice = EmptyLattice(lattice_constant)

    electron = FreeElectron(lattice, 1)
    
    mass = AtomicWeight.table["Li"] * Mass.DALTON["->"]
    
    debye_temperature = 344.0

    phonon = DebyeModel(lattice, debye_temperature, 1, mass)

    temperature = debye_temperature

    self_energy = SelfEnergy2nd(lattice, electron, phonon)

    n_g = np.array([1]*3)
    n_k = np.array([6]*3)
    g, k = electron.grid(n_g, n_k)
    
    n_g_inter = np.array([1]*3)
    n_q = np.array([10]*3)
    selfen = self_energy.calculate(temperature, g, k, n_g_inter, n_q)

    epsilon_nk = electron.eigenenergy(k + g)

    fig = plt.figure()

    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)

    for ax in [ax1, ax2]:
        ax.scatter(epsilon_nk * Energy.EV["<-"], selfen.imag / (Time.SI["<-"] / Prefix.PICO), label="$\mathrm{Im}\Sigma^\mathrm{Fan}$")

        ax.set_ylabel("Scattering rate ($\mathrm{ps}^{-1}$)")
        ax.legend()

    ax2.set_xlabel("Electron energy ($\mathrm{eV}$)")
    ax2.set_yscale("log")
    ax1.set_title("Example: Scattering rate of bcc-Li")
    
    file_name = "test_scattering_rate.png"
    fig.savefig(file_name)

if __name__ == "__main__":
    main()
```

## License
MIT