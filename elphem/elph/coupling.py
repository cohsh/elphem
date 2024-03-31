import numpy as np

from elphem.phonon.debye import DebyeModel
from elphem.elph.distribution import safe_divide

class Coupling:
    effective_potential: float = 1 / 16

    @classmethod
    def first_order(cls, g1: np.ndarray, g2: np.ndarray, q: np.ndarray, phonon: DebyeModel) -> np.ndarray:
        """
        Calculate lowest-order electron-phonon couplings.
        
        Args
            g1, g2: A numpy array representing G-vector
            k: A numpy array representing k-vector
            q: A numpy array representing k-vector
            phonon: a Debye model
        
        Return
            A value of the elctron-phonon coupling.
        """
        q_norm = np.linalg.norm(q, axis=-1)
        delta_g = g1 - g2
        q_dot = np.sum(q * delta_g, axis=-1) 

        mask = q_norm > 0
        result = np.zeros_like(q_norm)
        
        denominator = np.sqrt(2.0 * phonon.mass * phonon.speed) * q_norm ** 1.5
        result[mask] = safe_divide(cls.effective_potential * q_dot[mask], denominator[mask])
        
        return result