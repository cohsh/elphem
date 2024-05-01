from dataclasses import dataclass

class Couplings:
    effective_potential: float = 1.0 / 16.0

    def __init__(self, electron: FreeElectron, phonon: DebyePhonon):
        pass
    
    def calculate(self) -> np.ndarray:
        """Calculate the lowest-order electron-phonon coupling between states.

        Args:
            g1 (np.ndarray): Initial G-vector in reciprocal space.
            g2 (np.ndarray): Final G-vector in reciprocal space.
            q (np.ndarray): Phonon wave vector in reciprocal space.

        Returns:
            np.ndarray: The electron-phonon coupling strength for the given vectors.
        """
        
        couplings = -1.0j * self.effective_potential * np.sum((self.phonon.q + self.electron.g1 - self.electron.g2) * self.phonon.eigenvectors, axis=-1) * self.phonon.zero_point_lengths

        return couplings