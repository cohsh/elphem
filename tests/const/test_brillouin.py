from unittest import TestCase
from elphem.common.brillouin import SpecialPoints

class TestUnit(TestCase):
    def test_points(self):
        gamma = SpecialPoints.Gamma["G"]
        fcc_gamma = SpecialPoints.FCC["G"]
        bcc_h = SpecialPoints.BCC["H"]
        hexagonal_h = SpecialPoints.Hexagonal["H"]

        self.assertEqual(gamma, fcc_gamma)
        self.assertNotEqual(bcc_h, hexagonal_h)