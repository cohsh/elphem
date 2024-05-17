from unittest import TestCase
from elphem.common.brillouin import SpecialPoints3D, SpecialPoints2D, SpecialPoints1D

class TestUnit(TestCase):
    def test_3D_points(self):
        gamma = SpecialPoints3D.Gamma["G"]
        fcc_gamma = SpecialPoints3D.FCC["G"]
        bcc_h = SpecialPoints3D.BCC["H"]
        hexagonal_h = SpecialPoints3D.Hexagonal["H"]

        self.assertEqual(gamma, fcc_gamma)
        self.assertNotEqual(bcc_h, hexagonal_h)
    
    def test_2D_points(self):
        gamma = SpecialPoints2D.Gamma["G"]
        square_gamma = SpecialPoints2D.Square["G"]
        square_m = SpecialPoints2D.Square["M"]
        hexagonal_m = SpecialPoints2D.Hexagonal["M"]

        self.assertEqual(gamma, square_gamma)
        self.assertNotEqual(square_m, hexagonal_m)

    def test_1D_points(self):
        x = SpecialPoints1D.Line["X"]

        self.assertEqual(x, 0.5)