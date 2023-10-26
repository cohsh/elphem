import numpy as np
from scipy.spatial.transform import Rotation

class LatticeRotation:    
    @classmethod
    def optimize(cls, basis: np.ndarray, axis: np.ndarray) -> np.ndarray:
        """Optimize the rotation of the given basis to align with the specified axis."""
        basis_rotated = cls.match(basis, axis)
        basis_rotated = cls.search_posture(basis_rotated, axis)
        return basis_rotated

    @classmethod
    def match(cls, basis: np.ndarray, axis: np.ndarray) -> np.ndarray:
        """Match the given basis direction to the specified axis using quaternion rotation."""
        if np.linalg.norm(axis) == 0.0:
            raise ValueError("Axis vector should not be zero.")
        
        direction = cls.normalize(basis[0] + basis[1] + basis[2])
        n = cls.normalize(axis)

        cross = cls.normalize(np.cross(direction, n))
        dot = np.dot(direction, n)
        theta = np.arccos(dot) / 2.0
        sin_theta = np.sin(theta)

        quaternion = np.array([
            cross[0] * sin_theta,
            cross[1] * sin_theta,
            cross[2] * sin_theta,
            np.cos(theta)
        ])

        rotation = Rotation.from_quat(quaternion)
        basis_rotated = rotation.apply(basis)
        
        return basis_rotated

    @classmethod
    def around_axis(cls, axis: np.ndarray, v: np.ndarray, theta: float) -> np.ndarray:
        """Rotate vector v around the given axis by angle theta."""
        if np.linalg.norm(axis) == 0.0:
            raise ValueError("Axis vector should not be zero.")
        
        axis_dot_v = np.dot(axis, v)
        cos_theta = np.cos(theta)
        v_rotated = cos_theta * v
        v_rotated += np.sin(theta) * np.cross(axis, v)
        v_rotated += (1.0 - cos_theta) * axis_dot_v * axis

        return v_rotated

    @classmethod
    def search_posture(cls, basis: np.ndarray, axis: np.ndarray, angle_max: float = 360.0) -> np.ndarray:
        """Search for the best posture of the basis aligned with the given axis."""
        if np.linalg.norm(axis) == 0.0:
            raise ValueError("Axis vector should not be zero.")
        
        angle = np.linspace(0.0, np.radians(angle_max), 1000)
        s_min = 1.0e+100
        basis_searched = basis
        
        n = np.zeros(basis.shape)
        u = np.identity(3)
                    
        axis_normalized = cls.normalize(axis)
        for theta in angle:
            basis_rotated = cls.around_axis(axis_normalized, basis, theta)
            
            s = 0.0
            for i in range(3):
                j = i
                k = (i+1) % 3
                n[i] = np.cross(basis_rotated[j], basis_rotated[k])
                s += np.dot(n[i], u[i])
            
            if s < s_min:
                s_min = s
                basis_searched = basis_rotated

        return basis_searched

    @staticmethod
    def normalize(v: np.ndarray) -> np.ndarray:
        """Normalize the given vector."""
        v_norm = np.linalg.norm(v)

        if v_norm == 0.0:
            return v

        return v / v_norm