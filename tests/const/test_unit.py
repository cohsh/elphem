from unittest import TestCase

from elphem.const.unit import *

class TestUnit(TestCase):
    def test_byte(self):
        size = Byte.get_str(2**30)
        self.assertEqual(size, "1.0 GB")
    
    def test_length(self):
        self.assertEqual(Length.SI["<-"], Length.unit)
    
    def test_energy(self):
        self.assertEqual(Energy.SI["<-"], Energy.unit)
        self.assertEqual(Energy.eV["->"] * Energy.SI["<-"], 1.602176634e-19)
        self.assertEqual(Energy.kelvin["->"] * Energy.kelvin["<-"], 1.0)