# Elphem
[![Upload Python Package](https://github.com/cohsh/elphem/actions/workflows/python-publish.yml/badge.svg)](https://github.com/cohsh/elphem/actions/workflows/python-publish.yml)
[![Python package](https://github.com/cohsh/elphem/actions/workflows/python-package.yml/badge.svg)](https://github.com/cohsh/elphem/actions/workflows/python-package.yml)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/elphem)
![PyPI - Version](https://img.shields.io/pypi/v/elphem)
[![Downloads](https://static.pepy.tech/badge/elphem/month)](https://pepy.tech/project/elphem)
![GitHub](https://img.shields.io/github/license/cohsh/elphem)

**El**ectron-**Ph**onon Interactions with **Em**pty Lattice

- PyPI: https://pypi.org/project/elphem

## Installation
### From PyPI
```shell
pip install elphem
```

### From GitHub
```shell
git clone git@github.com:cohsh/elphem.git
cd elphem
pip install -e .
```

## Features
Currently, Elphem allows calculations of
- (reciprocal) lattice vectors from lattice constants.
- electronic structures with empty lattice approximation.
- phonon dispersion relations with Debye model.
- first-order electron-phonon couplings.
- one-electron self-energies.
- spectral functions.

## Examples
### Calculation of spectral functions (`examples/spectrum.py`)
![spectrum](images/spectrum.png)

```python
"""Example: bcc-Li"""
import numpy as np
import matplotlib.pyplot as plt
from elphem import *

def main():
    a = 2.98 * Length.ANGSTROM["->"]
    mass = AtomicWeight.table["Li"] * Mass.DALTON["->"]

    debye_temperature = 344.0

    lattice = EmptyLattice('bcc', a)
    electron = FreeElectron(lattice, n_band=8, n_electron=1)
    phonon = DebyeModel(lattice, debye_temperature, 1, mass)

    temperature =  3 * debye_temperature
    self_energy = SelfEnergy(lattice, electron, phonon, temperature, sigma=0.5, eta=0.1)

    n_q = np.array([10]*3)
    n_omega = 1000
    range_omega = [-8 * Energy.EV["->"], 6 * Energy.EV["->"]]
    
    k_names = ["G", "H", "N", "G", "P", "H"]
    n_split = 20
    
    x, y, spectrum, special_x = Spectrum(self_energy).calculate_with_path(k_names, n_split, n_q, n_omega, range_omega)
    y_mesh, x_mesh = np.meshgrid(y, x)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    
    mappable = ax.pcolormesh(x_mesh, y_mesh * Energy.EV["<-"], spectrum / Energy.EV["<-"])
    
    for x0 in special_x:
        ax.axvline(x=x0, color="black", linewidth=0.3)
    
    ax.set_xticks(special_x)
    ax.set_xticklabels(k_names)
    ax.set_ylabel("Energy ($\mathrm{eV}$)")
    ax.set_title("Spectral function of bcc-Li")
    
    fig.colorbar(mappable, ax=ax)

    fig.savefig("example_spectrum.png")

if __name__ == "__main__":
    main()
```

### Calculation of the electron-phonon renormalization (EPR) (`examples/electron_phonon_renormalization.py`)

![epr](images/epr.png)

```python
"""Example: bcc-Li"""
import numpy as np
import matplotlib.pyplot as plt
from elphem import *

def main():
    a = 2.98 * Length.ANGSTROM["->"]
    mass = AtomicWeight.table["Li"] * Mass.DALTON["->"]
    debye_temperature = 344.0
    temperature = 3 * debye_temperature
    n_band = 20

    lattice = EmptyLattice('bcc', a)
    electron = FreeElectron(lattice, n_band, 1)        
    phonon = DebyeModel(lattice, temperature, 1, mass)

    self_energy = SelfEnergy(lattice, electron, phonon, temperature, eta=0.05)

    k_names = ["G", "H", "N", "G", "P", "H"]

    n_split = 20
    n_q = np.array([8]*3)
    
    k, eig, epr, special_k = EPR(self_energy).calculate_with_path(k_names, n_split, n_q)
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    for n in range(n_band):
        if n == 0:
            ax.plot(k, eig[n] * Energy.EV["<-"], color="tab:blue", label="w/o EPR")
            ax.plot(k, (eig[n] + epr[n]) * Energy.EV["<-"], color="tab:orange", label="w/ EPR")
        else:
            ax.plot(k, eig[n] * Energy.EV["<-"], color="tab:blue")
            ax.plot(k, (eig[n] + epr[n]) * Energy.EV["<-"], color="tab:orange")
    
    for k0 in special_k:
        ax.axvline(x=k0, color="black", linewidth=0.3)
    
    ax.set_xticks(special_k)
    ax.set_xticklabels(k_names)
    ax.set_ylabel("Energy ($\mathrm{eV}$)")
    ax.set_title("Example: Band structure of bcc-Li")
    ax.set_ylim([-10,20])
    ax.legend()


    fig.savefig("example_epr.png")

if __name__ == "__main__":
    main()
```

### Calculation of the electronic band structure (`examples/band_structure.py`)

![band structure](images/band_structure.png)

```python
"""Example: bcc-Li"""
import matplotlib.pyplot as plt
from elphem import *

def main():
    a = 2.98 * Length.ANGSTROM["->"]

    lattice = EmptyLattice('bcc', a)
    electron = FreeElectron(lattice, n_band=50, n_electron=1)

    k_names = ["G", "H", "N", "G", "P", "H"]

    k, eig, special_k = electron.get_band_structure(k_names, n_split=20)

    fig, ax = plt.subplots()
    for band in eig:
        ax.plot(k, band * Energy.EV["<-"], color="tab:blue")
    
    ax.vlines(special_k, ymin=-10, ymax=50, color="black", linewidth=0.3)
    ax.set_xticks(special_k)
    ax.set_xticklabels(k_names)
    ax.set_ylabel("Energy ($\mathrm{eV}$)")
    ax.set_ylim([-10,50])

    fig.savefig("example_band_structure.png")

if __name__ == "__main__":
    main()
```

### Calculation of the phonon dispersion (`examples/phonon_dispersion.py`)

![phonon dispersion](images/phonon_dispersion.png)

```python
"""Example: bcc-Li"""
import matplotlib.pyplot as plt
from elphem import *

def main():
    a = 2.98 * Length.ANGSTROM["->"]
    mass = AtomicWeight.table["Li"] * Mass.DALTON["->"]
    lattice = EmptyLattice('bcc', a)

    debye_temperature = 344.0
    phonon = DebyeModel(lattice, debye_temperature, 1, mass)

    q_names = ["G", "H", "N", "G", "P", "H"]
    
    q, omega, special_q = phonon.get_dispersion(q_names, n_split=20)
    
    fig, ax = plt.subplots()

    ax.plot(q, omega * Energy.EV["<-"] * 1.0e+3, color="tab:blue")
    
    for q0 in special_q:
        ax.axvline(x=q0, color="black", linewidth=0.3)
    
    ax.set_xticks(special_q)
    ax.set_xticklabels(q_names)
    ax.set_ylabel("Energy ($\mathrm{meV}$)")

    fig.savefig("example_phonon_dispersion.png")

if __name__ == "__main__":
    main()
```

## License
MIT

## Author
Kohei Ishii