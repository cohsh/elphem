from types import MappingProxyType

class SpecialPoints3D:
    """Defines immutable sets of special points in first Brillouin zone for different types of crystal lattices.

    Attributes:
        Gamma (MappingProxyType): Immutable set containing the Gamma point.
        SC (MappingProxyType): Immutable set of special points for the simple cubic lattice.
        FCC (MappingProxyType): Immutable set of special points for the face-centered cubic lattice.
        BCC (MappingProxyType): Immutable set of special points for the body-centered cubic lattice.
        Hexagonal (MappingProxyType): Immutable set of special points for the hexagonal lattice.
    """
    # Gamma Point
    _Gamma = {
        "G": (0, 0, 0)
    }
    Gamma = MappingProxyType(_Gamma)

    # Points of Each Lattice
    _SC = {**_Gamma}
    _SC.update({
        "R": (1/2, 1/2, 1/2),
        "X": (0, 1/2, 0),
        "M": (1/2, 1/2, 0)
    })
    SC = MappingProxyType(_SC)

    _FCC = {**_Gamma}
    _FCC.update({
        "X": (0, 1/2, 1/2),
        "L": (1/2, 1/2, 1/2),
        "W": (1/4, 3/4, 1/2),
        "U": (1/4, 5/8, 5/8),
        "K": (3/8, 3/4, 3/8)
    })
    FCC = MappingProxyType(_FCC)

    _BCC = {**_Gamma}
    _BCC.update({
        "H": (-1/2, 1/2, 1/2),
        "P": (1/4, 1/4, 1/4),
        "N": (0, 1/2, 0)
    })
    BCC = MappingProxyType(_BCC)

    _Hexagonal = {**_Gamma}
    _Hexagonal.update({
        "A": (0, 0, 1/2),
        "K": (2/3, 1/3, 0),
        "H": (2/3, 1/3, 1/2),
        "M": (1/2, 0, 0),
        "L": (1/2, 0, 1/2)
    })
    Hexagonal = MappingProxyType(_Hexagonal)

class SpecialPoints2D:
    # Gamma Point
    _Gamma = {
        "G": (0, 0)
    }
    Gamma = MappingProxyType(_Gamma)

    _Square = {**_Gamma}
    _Square.update({
        "X": (1/2, 0),
        "M": (1/2, 1/2)
    })
    Square = MappingProxyType(_Square)
    
    _Hexagonal = {**_Gamma}
    _Hexagonal.update({
        "M": (1/2, 0),
        "K": (1/2, 1/2)
    })
    Hexagonal = MappingProxyType(_Hexagonal)

class SpecialPoints1D:
    # Gamma Point
    _Gamma = {
        "G": 0
    }
    Gamma = MappingProxyType(_Gamma)
    
    _Line = {**_Gamma}
    _Line.update({
        "X": 1/2
    })
    Line = MappingProxyType(_Line)