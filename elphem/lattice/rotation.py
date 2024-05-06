import numpy as np
from scipy.spatial.transform import Rotation

class LatticeRotation:
    @staticmethod
    def normalize(v: np.ndarray) -> np.ndarray:
        """Normalize the given vector.

        Args:
            v (np.ndarray): The vector to normalize.

        Returns:
            np.ndarray: The normalized vector.

        Raises:
            ValueError: If the input vector is zero.
        """
        v_norm = np.linalg.norm(v)

        if v_norm == 0.0:
            return v

        return v / v_norm

class LatticeRotation3D(LatticeRotation):
    """Provides methods to optimize and adjust lattice orientations through rotation operations."""
    
    @classmethod
    def calculate_optimized_basis(cls, basis: np.ndarray, axis: np.ndarray) -> np.ndarray:
        """Optimize the rotation of the given basis to align with the specified axis.

        Args:
            basis (np.ndarray): The basis vectors of the lattice to be rotated.
            axis (np.ndarray): The target axis for alignment.

        Returns:
            np.ndarray: The optimized basis vectors aligned as closely as possible to the target axis.

        Raises:
            ValueError: If the axis vector is zero.
        """
        basis_rotated = cls.search_optimized_direction(basis, axis)
        basis_rotated = cls.search_optimized_posture(basis_rotated, axis)
        return basis_rotated

    @classmethod
    def search_optimized_direction(cls, basis: np.ndarray, axis: np.ndarray) -> np.ndarray:
        """search_optimized_direction the given basis direction to the specified axis using quaternion rotation.

        Args:
            basis (np.ndarray): The initial basis vectors.
            axis (np.ndarray): The axis to align the basis vectors to.

        Returns:
            np.ndarray: The basis vectors rotated to search_optimized_direction the direction of the axis.

        Raises:
            ValueError: If the axis vector is zero.
        """
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
    def rotate_vector_around_axis(cls, axis: np.ndarray, v: np.ndarray, theta: float) -> np.ndarray:
        """Rotate vector v around the given axis by angle theta.

        Args:
            axis (np.ndarray): The axis of rotation.
            v (np.ndarray): The vector to rotate.
            theta (float): The angle in radians to rotate the vector.

        Returns:
            np.ndarray: The rotated vector.

        Raises:
            ValueError: If the axis vector is zero.
        """
        if np.linalg.norm(axis) == 0.0:
            raise ValueError("Axis vector should not be zero.")
        
        axis_dot_v = np.dot(axis, v)
        cos_theta = np.cos(theta)
        v_rotated = cos_theta * v
        v_rotated += np.sin(theta) * np.cross(axis, v)
        v_rotated += (1.0 - cos_theta) * axis_dot_v * axis

        return v_rotated

    @classmethod
    def search_optimized_posture(cls, basis: np.ndarray, axis: np.ndarray, angle_max: float = 360.0) -> np.ndarray:
        """Search for the best posture of the basis aligned with the given axis.

        Args:
            basis (np.ndarray): The basis to be rotated.
            axis (np.ndarray): The axis to align with.
            angle_max (float): The maximum angle range to search in degrees.

        Returns:
            np.ndarray: The optimally aligned basis after posture search.

        Raises:
            ValueError: If the axis vector is zero.
        """
        if np.linalg.norm(axis) == 0.0:
            raise ValueError("Axis vector should not be zero.")
        
        angle = np.linspace(0.0, np.radians(angle_max), 1000)
        s_min = 1.0e+100
        basis_searched = basis
        
        n = np.zeros(basis.shape)
        u = np.identity(3)
                    
        axis_normalized = cls.normalize(axis)
        for theta in angle:
            basis_rotated = cls.rotate_vector_around_axis(axis_normalized, basis, theta)
            
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

class LatticeRotation2D(LatticeRotation):
    """Provides methods to optimize and adjust lattice orientations through rotation operations."""
    
    @classmethod
    def calculate_optimized_basis(cls, basis: np.ndarray, axis: np.ndarray) -> np.ndarray:
        """Optimize the rotation of the given basis to align with the specified axis.

        Args:
            basis (np.ndarray): The basis vectors of the lattice to be rotated.
            axis (np.ndarray): The target axis for alignment.

        Returns:
            np.ndarray: The optimized basis vectors aligned as closely as possible to the target axis.

        Raises:
            ValueError: If the axis vector is zero.
        """
        basis_rotated = cls.search_optimized_direction(basis, axis)
        return basis_rotated

    @classmethod
    def search_optimized_direction(cls, basis: np.ndarray, axis: np.ndarray) -> np.ndarray:
        """search_optimized_direction the given basis direction to the specified axis using quaternion rotation.

        Args:
            basis (np.ndarray): The initial basis vectors.
            axis (np.ndarray): The axis to align the basis vectors to.

        Returns:
            np.ndarray: The basis vectors rotated to search_optimized_direction the direction of the axis.

        Raises:
            ValueError: If the axis vector is zero.
        """
        if np.linalg.norm(axis) == 0.0:
            raise ValueError("Axis vector should not be zero.")
        
        direction = cls.normalize(basis[0] + basis[1])
        n = cls.normalize(axis)

        dot = np.dot(direction, n)

        theta = np.arccos(dot)
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        
        rotation_matrix = np.array([
            [cos_theta, -sin_theta],
            [sin_theta, cos_theta]
        ])

        basis_rotated = rotation_matrix @ basis
        
        return basis_rotated