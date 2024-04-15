import numpy as np
from dataclasses import dataclass

from elphem.lattice.rotation import LatticeRotation
from elphem.const.brillouin import SpecialPoints

@dataclass
class LatticeConstant:
    a: float
    b: float
    c: float
    alpha: float
    beta: float
    gamma: float
    crystal_structure: str
    
    def __post_init__(self):
        self.length = np.array([self.a, self.b, self.c])
        self.angle = np.radians(np.array([self.alpha, self.beta, self.gamma]))
        
    def rescale(self, factor: float) -> None:
        self.length *= factor

class Cell:
    def __init__(self):
        self.basis = self.build()
    
    def build(self) -> np.ndarray:
        return np.identity(3)
    
    def volume(self) -> float:
        volume = np.dot(self.basis[0], np.cross(self.basis[1], self.basis[2]))
        return volume
    
    def element(self, a: np.ndarray) -> np.ndarray:
        x = np.dot(a, self.basis)
        return x

    @staticmethod
    def optimize(basis: np.ndarray) -> np.ndarray:
        axis = np.array([1.0] * 3)
        
        basis = LatticeRotation.optimize(basis, axis)
        
        return basis

@dataclass
class PrimitiveCell(Cell):
    lattice_constant: LatticeConstant
    
    def __post_init__(self):
        self.basis = self.build()
    
    def build(self) -> np.ndarray:
        length = self.lattice_constant.length
        angle = self.lattice_constant.angle

        basis = np.zeros((3,3))

        basis[0][0] = length[0]
        basis[1][0] = length[1] * np.cos(angle[2])
        basis[1][1] = length[1] * np.sin(angle[2])

        basis[2][0] = length[2] * np.cos(angle[1])
        basis[2][1] = length[2] * (np.cos(angle[0]) - np.cos(angle[1]) * np.cos(angle[2])) / np.sin(angle[2])
        basis[2][2] = np.sqrt(length[2] ** 2 - np.sum(basis[2]**2))

        basis = self.optimize(basis)
        
        return basis

@dataclass
class ReciprocalCell(Cell):
    lattice_constant: LatticeConstant
    
    def __post_init__(self):
        self.basis = self.build()
    
    def build(self) -> np.ndarray:
        primitive_cell = PrimitiveCell(self.lattice_constant)

        basis = np.zeros((3,3))
        
        primitive_vector = primitive_cell.basis
        for i in range(3):
            j = (i+1) % 3
            k = (i+2) % 3
            basis[i] = np.cross(primitive_vector[j], primitive_vector[k])

        basis *= 2.0 * np.pi / primitive_cell.volume()
        
        return basis

    def path(self, k_names: list[str], n: int) -> np.ndarray:
        # k_names = ["G", "H", "N", "G", "P", "H"]
        k_via = [self.get_special_k(s) for s in k_names]

        n_via = len(k_via) - 1

        total_length = np.empty((n_via * n,))
        special_length = np.empty((n_via+1,))
        k = np.empty((n_via * n, 3))

        count = 0
        length_part = 0.0
        special_length[0] = 0.0

        for i in range(n_via):
            direction = (np.array(k_via[i+1]) - np.array(k_via[i])) @ self.basis
            length = np.linalg.norm(direction)

            x = np.linspace(0.0, 1.0, n)
                        
            for j in range(n):
                k[count] = k_via[i] @ self.basis + x[j] * direction
                total_length[count] = x[j] * length + length_part
                count += 1
            
            length_part += length
            special_length[i+1] = length_part

        return total_length, k, special_length

    def get_special_k(self, k_name: str) -> np.ndarray:
        if self.lattice_constant.crystal_structure == 'bcc':
            return SpecialPoints.BCC[k_name]
        elif self.lattice_constant.crystal_structure == 'fcc':
            return SpecialPoints.FCC[k_name]
        elif self.lattice_constant.crystal_structure == 'sc':
            return SpecialPoints.SC[k_name]
        else:
            raise ValueError("Invalid name specified.")

@dataclass
class EmptyLattice:
    crystal_structure: str
    a: float

    def __post_init__(self):
        self.constants = self.get_lattice_constant()
        self.primitive_cell = PrimitiveCell(self.constants)
        self.reciprocal_cell = ReciprocalCell(self.constants)

        self.volume = {
            "primitive": self.primitive_cell.volume(),
            "reciprocal": self.reciprocal_cell.volume()
        }
        self.basis = {
            "primitive": self.primitive_cell.basis,
            "reciprocal": self.reciprocal_cell.basis
        }

    def get_lattice_constant(self) -> LatticeConstant:
        crystal_structure_lower = self.crystal_structure.lower()

        alpha_values = {
            'bcc': 109.47,
            'fcc': 60.0,
            'sc': 90.0
        }

        alpha = alpha_values.get(crystal_structure_lower)

        if alpha is not None:
            return LatticeConstant(self.a, self.a, self.a, alpha, alpha, alpha, crystal_structure_lower)
        else:
            raise ValueError("Invalid crystal structure specified.")
        
    def grid(self, *n: list[np.ndarray], space="reciprocal") -> np.ndarray:
        basis = self.basis[space]
        
        n_array = np.array(n)
        n_point = len(n_array)
        n_array = n_array.reshape(n_array.size,)

        grid = np.meshgrid(*[np.arange(-i, i) for i in n_array])
        grid = np.array(grid)

        grid_set = []
        j = 0
        for i in range(n_point):
            x = grid[j:j+3]
            y = np.empty(x[0].shape + (3,))
            for k in range(3):
                y[...,k] = x[k]

            grid_set.append(y @ basis)
            j += 3

        if len(grid_set) == 1:
            return grid_set[0]
        else:
            return tuple(grid_set)