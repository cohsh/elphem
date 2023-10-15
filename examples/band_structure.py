import numpy as np

from elphem import BCC, Energy, Length, LatticeConstant, EmptyLattice, FreeElectron

def main():
    # Example: Li (BCC)
    a = 2.98 * Length.angstrom["->"]
    alpha = 109.47

    lattice_constant = LatticeConstant(a,a,a,alpha,alpha,alpha)
    lattice = EmptyLattice(lattice_constant)

    n_cut = np.array([2]*3)
    electron = FreeElectron(lattice, 1)
        
    k_names = ["G", "H", "N", "G", "P", "H"]

    k_via = []
    for name in k_names:
        k_via.append(BCC.points[name])
    
    file_name = "test_band_structure.png"
    
    electron.save_band(file_name, n_cut, k_names, *k_via, ylim=[-6.0, Energy.eV["<-"]])

if __name__ == "__main__":
    main()