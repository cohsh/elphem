import math
import scipy.constants as constants

class Prefix:
    """ Metric prefixes. """
    EXA = 1.0e+18
    PETA = 1.0e+15
    TERA = 1.0e+12
    GIGA = 1.0e+9
    MEGA = 1.0e+6
    KILO = 1.0e+3
    HECTO = 1.0e+2
    DECA = 1.0e+1
    DECI = 1.0e-1
    CENTI = 1.0e-2
    MILLI = 1.0e-3
    MICRO = 1.0e-6
    NANO = 1.0e-9
    PICO = 1.0e-12
    FEMTO = 1.0e-15
    ATTO = 1.0e-18

class Byte:
    """ Digital information units based on powers of 2. """
    KILO = 2 ** 10
    MEGA = 2 ** 20
    GIGA = 2 ** 30
    TERA = 2 ** 40
    PETA = 2 ** 50
    EXA = 2 ** 60

    @staticmethod
    def get_str(size: int) -> str:
        """Returns a human-readable string representation of digital information sizes, scaled to appropriate unit.

        Args:
            size (int): Size in bytes.

        Returns:
            str: Formatted string representing the size in appropriate unit (e.g., KB, MB).

        Raises:
            ValueError: If size is negative or too large to be represented.
        """
        if size < 0:
            raise ValueError("Size should be a non-negative integer.")
        
        units = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB")
        i = math.floor(math.log(size, 1024)) if size > 0 else 0
        
        if i >= len(units):
            raise ValueError("Size too large to represent in known units.")
        
        size = round(size / 1024 ** i, 2)
        return f"{size} {units[i]}"

class AtomicUnits:
    """ Utility class for atomic unit conversions. """
    @staticmethod
    def convert(unit: float, value: float) -> dict:
        """Converts a value to and from a base unit to facilitate comparisons and calculations.

        Args:
            unit (float): The unit to which the conversion is based.
            value (float): The value to be converted.

        Returns:
            dict: A dictionary containing converted values.
        """
        return {
            "->": value / unit,
            "<-": unit / value
        }

class Length:
    """ Atomic units of length. """
    UNIT = 5.29177210903e-11
    SI = AtomicUnits.convert(UNIT, 1.0)
    ANGSTROM = AtomicUnits.convert(UNIT, 1.0e-10)

class Energy:
    """ Atomic units of energy. """
    UNIT = 4.3597447222071e-18
    SI = AtomicUnits.convert(UNIT, 1.0)
    EV = AtomicUnits.convert(UNIT, constants.eV)
    KELVIN = AtomicUnits.convert(UNIT, constants.k)

class Mass:
    """ Atomic units of mass. """
    UNIT = 9.1093837015e-31
    SI = AtomicUnits.convert(UNIT, 1.0)
    DALTON = AtomicUnits.convert(UNIT, 1.66053906660e-27)
    PROTON = AtomicUnits.convert(UNIT, 1.672621898e-27)

class Time:
    """ Atomic units of time. """
    UNIT = 2.4188843265857e-17
    SI = AtomicUnits.convert(UNIT, 1.0)

class Velocity:
    """ Atomic units of velocity. """
    UNIT = 2.18769126364e+6
    SI = AtomicUnits.convert(UNIT, 1.0)

class ElectricPotential:
    """ Atomic units of electric potential. """
    UNIT = 27.211386245988
    SI = AtomicUnits.convert(UNIT, 1.0)