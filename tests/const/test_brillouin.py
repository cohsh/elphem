from unittest import TestCase

from elphem.const.brillouin import *

class TestUnit(TestCase):
    def test_points(self):
        self.assertEqual(FCC.points["G"], Gamma.points["G"])
        self.assertNotEqual(BCC.points["H"], Hexagonal.points["H"])