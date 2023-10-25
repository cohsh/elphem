class AtomicWeight:
    table = {
    "H": 1.008,
    "He": 4.002602,
    "Li": 6.94,
    "Be": 9.0121831,
    "B": 10.81,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "F": 18.99840316,
    "Ne": 20.1797,
    "Na": 22.98976928,
    "Mg": 24.305,
    "Al": 26.9815384,
    "Si": 28.085,
    "P": 30.973762,
    "S": 32.06,
    "Cl": 35.45,
    "Ar": 39.95,
    "K": 39.0983,
    "Ca": 40.078,
    "Sc": 44.955907,
    "Ti": 47.867,
    "V": 50.9415,
    "Cr": 51.9961,
    "Mn": 54.9380432,
    "Fe": 55.8452,
    "Co": 58.9331943,
    "Ni": 58.69344,
    "Cu": 63.5463,
    "Zn": 65.382,
    "Ga": 69.7231,
    "Ge": 72.6308,
    "As": 74.9215956,
    "Se": 78.9718,
    "Br": 79.904,
    "Kr": 83.7982,
    "Rb": 85.46783,
    "Sr": 87.621,
    "Y": 88.9058382,
    "Zr": 91.2242,
    "Nb": 92.906371,
    "Mo": 95.951,
    "Tc": 97,
    "Ru": 101.072,
    "Rh": 102.905492,
    "Pd": 106.421,
    "Ag": 107.86822,
    "Cd": 112.4144,
    "In": 114.8181,
    "Sn": 118.7107,
    "Sb": 121.7601,
    "Te": 127.603,
    "I": 126.904473,
    "Xe": 131.2936,
    "Cs": 132.905451966,
    "Ba": 137.3277,
    "La": 138.905477,
    "Ce": 140.1161,
    "Pr": 140.907661,
    "Nd": 144.2423,
    "Pm": 145,
    "Sm": 150.362,
    "Eu": 151.9641,
    "Gd": 157.253,
    "Tb": 158.9253547,
    "Dy": 162.5001,
    "Ho": 164.9303295,
    "Er": 167.2593,
    "Tm": 168.9342195,
    "Yb": 173.04510,
    "Lu": 174.96681,
    "Hf": 178.4866,
    "Ta": 180.947882,
    "W": 183.841,
    "Re": 186.2071,
    "Os": 190.233,
    "Ir": 192.2172,
    "Pt": 195.0849,
    "Au": 196.9665704,
    "Hg": 200.5923,
    "Tl": 204.38
    }
    
    @classmethod
    def get_from_list(cls, atomic_names: list) -> list:
        if not isinstance(atomic_names, list):
            raise TypeError("Input must be a list of atomic symbols.")
            
        mass_list = []
        for name in atomic_names:
            if not isinstance(name, str):
                raise TypeError("Each item in the list must be a string representing an atomic symbol.")
                
            try:
                atomic_mass = cls.table[name]
            except KeyError:
                raise ValueError(f"{name} is not a valid atomic symbol.")
            else:
                mass_list.append(atomic_mass)
        return mass_list