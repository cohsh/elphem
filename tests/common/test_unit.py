from unittest import TestCase
from elphem.common.unit import *

class TestUnit(TestCase):
    def test_byte(self):
        size = Byte.get_str(2**30)
        self.assertEqual(size, "1.0 GB")
    
    def test_length(self):
        self.assertEqual(Length.SI["<-"], Length.UNIT)
    
    def test_energy(self):
        self.assertEqual(Energy.SI["<-"], Energy.UNIT)
        self.assertEqual(Energy.EV["->"] * Energy.SI["<-"], 1.602176634e-19)
        self.assertEqual(Energy.KELVIN["->"] * Energy.KELVIN["<-"], 1.0)