# elphem
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
- direct and reciprocal lattice vectors from lattice constants with optimization.
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
    # Parameters of lattice
    a = 2.98 * Length.ANGSTROM['->']

    # Parameters of electron
    n_electrons = 1
    n_bands_electron = 4

    # Parameters of phonon
    debye_temperature = 344.0
    n_q = [6, 6, 6]
    
    # Parameters of k-path
    k_names = ["G", "H", "N", "G", "P", "H"]
    n_split = 50
    
    # Parameters of electron-phonon
    temperature = 0.8 * debye_temperature
    n_bands_elph = 4

    # Generate a lattice
    lattice = Lattice3D('bcc', 'Li', a)

    # Get k-path
    k_path = lattice.get_k_path(k_names, n_split)

    # Generate an electron.
    electron = Electron.create_from_path(lattice, n_electrons, n_bands_electron, k_path)

    # Generate a phonon.
    phonon = Phonon.create_from_n(lattice, debye_temperature, n_q)

    # Generate electron-phonon
    electron_phonon = ElectronPhonon(electron, phonon, temperature, n_bands_elph)

    # Set frequencies
    n_omega = 200
    range_omega = [-6 * Energy.EV["->"], 20 * Energy.EV["->"]]
    omega_array = np.linspace(range_omega[0] , range_omega[1], n_omega)
    
    # Calculate a spectral function
    spectrum = electron_phonon.calculate_spectrum_over_range(omega_array, normalize=True)
    
    y, x = np.meshgrid(omega_array, k_path.minor_scales)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    mappable = ax.pcolormesh(x, y * Energy.EV["<-"], spectrum / Energy.EV["<-"])
    
    for x0 in k_path.major_scales:
        ax.axvline(x=x0, color="black", linewidth=0.3)
    
    ax.set_xticks(k_path.major_scales)
    ax.set_xticklabels(k_names)
    
    ax.set_ylabel("Energy ($\mathrm{eV}$)")
    ax.set_title("Spectral function of bcc-Li (Normalized)")
    
    fig.colorbar(mappable, ax=ax)
    mappable.set_clim(0.00, 0.03)

    fig.savefig("spectrum.png")

if __name__ == "__main__":
    main()
```

### Calculation of the electron-phonon renormalization (EPR) (`examples/epr.py`)

![epr](images/epr.png)

```python
"""Example: bcc-Li"""
import numpy as np
import matplotlib.pyplot as plt
from elphem import *

def main():
    # Parameters of lattice
    a = 2.98 * Length.ANGSTROM['->']

    # Parameters of electron
    n_electrons = 1
    n_bands_electron = 20

    # Parameters of phonon
    debye_temperature = 344.0
    n_q = [10, 10, 10]
    
    # Parameters of k-path
    k_names = ["G", "H", "N", "G", "P", "H"]
    n_split = 20
    
    # Parameters of electron-phonon
    temperature = 300.0
    n_bands_elph = 1

    # Generate a lattice
    lattice = Lattice3D('bcc', 'Li', a)

    # Get k-path
    k_path = lattice.get_k_path(k_names, n_split)

    # Generate an electron.
    electron = Electron.create_from_path(lattice, n_electrons, n_bands_electron, k_path)

    # Generate a phonon.
    phonon = Phonon.create_from_n(lattice, debye_temperature, n_q)

    # Generate electron-phonon
    electron_phonon = ElectronPhonon(electron, phonon, temperature, n_bands_elph, eta=0.05)
    
    # Calculate electron-phonon renormalization
    epr = electron_phonon.calculate_electron_phonon_renormalization()
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    for i in range(n_bands_elph):
        ax.plot(k_path.minor_scales, electron.eigenenergies[i] * Energy.EV["<-"], color='tab:blue', label='w/o EPR')
        ax.plot(k_path.minor_scales, (electron.eigenenergies[i] + epr[i]) * Energy.EV["<-"], color='tab:orange', label='w/ EPR')
    
    for x0 in k_path.major_scales:
        ax.axvline(x=x0, color="black", linewidth=0.3)

    ax.legend()
    
    ax.set_xticks(k_path.major_scales)
    ax.set_xticklabels(k_names)
    
    ax.set_ylabel("Energy ($\mathrm{eV}$)")
    ax.set_title("EPR of bcc-Li ($T=300~\mathrm{K}$)")

    fig.savefig("epr.png")

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
    a = 2.98 * Length.ANGSTROM['->']
    n_electrons = 1
    n_bands = 20

    lattice = Lattice3D('bcc', 'Li', a)
    k_names = ["G", "H", "N", "G", "P", "H"]
    
    k_path = lattice.reciprocal.get_path(k_names, 100)

    electron = Electron.create_from_path(lattice, n_electrons, n_bands, k_path)

    eigenenergies = electron.eigenenergies * Energy.EV['<-']

    fig, ax = plt.subplots()
    for band in eigenenergies:
        ax.plot(k_path.minor_scales, band, color="tab:blue")
    
    y_range = [-10, 50]
    
    ax.vlines(k_path.major_scales, ymin=y_range[0], ymax=y_range[1], color="black", linewidth=0.3)
    ax.set_xticks(k_path.major_scales)
    ax.set_xticklabels(k_names)
    ax.set_ylabel("Energy ($\mathrm{eV}$)")
    ax.set_ylim(y_range)

    fig.savefig("band_structure.png")

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
    lattice = Lattice3D('bcc', 'Li', a)

    q_names = ["G", "H", "N", "G", "P", "H"]
    q_path = lattice.reciprocal.get_path(q_names, 40)

    debye_temperature = 344.0
    phonon = Phonon.create_from_path(lattice, debye_temperature, q_path)
    
    omega = phonon.eigenenergies
    
    fig, ax = plt.subplots()

    ax.plot(q_path.minor_scales, omega * Energy.EV["<-"] * 1.0e+3, color="tab:blue")
    
    for q0 in q_path.major_scales:
        ax.axvline(x=q0, color="black", linewidth=0.3)
    
    ax.set_xticks(q_path.major_scales)
    ax.set_xticklabels(q_names)
    ax.set_ylabel("Energy ($\mathrm{meV}$)")

    fig.savefig("phonon_dispersion.png")

if __name__ == "__main__":
    main()
```

## License
MIT

## Author
Kohei Ishii (The University of Tokyo, Japan)