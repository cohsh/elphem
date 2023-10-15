class Gamma(dict):
    points = {
        "G" : [0, 0, 0]
        }

class SC(Gamma):
    points = Gamma.points | {
    "R": [1/2, 1/2, 1/2],
    "X": [0, 1/2, 0],
    "M": [1/2, 1/2, 0]        
    }

class FCC(Gamma):
    points = Gamma.points | {
    "X": [0, 1/2, 1/2],
    "L": [1/2, 1/2, 1/2],
    "W": [1/4, 3/4, 1/2],
    "U": [1/4, 5/8, 5/8],
    "K": [3/8, 3/4, 3/8]        
    }

class BCC(Gamma):
    points = Gamma.points | {
    "H": [-1/2, 1/2, 1/2],
    "P": [1/4, 1/4, 1/4],
    "N": [0, 1/2, 0]   
    }

class Hexagonal(Gamma):
    points = Gamma.points | {
    "A": [0, 0, 1/2],
    "K": [2/3, 1/3, 0],
    "H": [2/3, 1/3, 1/2],
    "M": [1/2, 0, 0],
    "L": [1/2, 0, 1/2]        
    }