import math
import scipy.constants as constants

class Prefix:
    exa = 1.0e+18
    peta = 1.0e+15
    tera = 1.0e+12
    giga = 1.0e+9
    mega = 1.0e+6
    kilo = 1.0e+3
    hecto = 1.0e+2
    deca = 1.0e+1
    deci = 1.0e-1
    centi = 1.0e-2
    milli = 1.0e-3
    micro = 1.0e-6
    nano = 1.0e-9
    pico = 1.0e-12
    femto = 1.0e-15
    atto = 1.0e-18

class Byte:
    kilo = 2 ** 10
    mega = 2 ** 20
    giga = 2 ** 30
    tera = 2 ** 40
    peta = 2 ** 50

    @staticmethod
    def get_str(size: int) -> str:
        units = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB")
        i = math.floor(math.log(size, 1024)) if size > 0 else 0
        size = round(size / 1024 ** i, 2)

        return f"{size} {units[i]}"

class AtomicUnits:
    @staticmethod
    def convert(unit: float, value: float) -> dict:
        converted_value = {
            "->": value / unit,
            "<-": unit / value
        }
        return converted_value

class Length(AtomicUnits):
    unit = 5.29177210903e-11
    SI: dict = AtomicUnits.convert(unit, 1.0)
    angstrom: dict = AtomicUnits.convert(unit, 1.0e-10)

class Energy(AtomicUnits):
    unit: float = 4.3597447222071e-18
    SI: dict = AtomicUnits.convert(unit, 1.0)
    eV: dict = AtomicUnits.convert(unit, constants.eV)
    kelvin: dict = AtomicUnits.convert(unit, constants.k)

class Mass(AtomicUnits):
    unit: float = 9.1093837015e-31
    SI: dict = AtomicUnits.convert(unit, 1.0)
    Dalton: dict = AtomicUnits.convert(unit, 1.66053906660e-27)
    proton: dict = AtomicUnits.convert(unit, 1.672621898e-27)

class Time(AtomicUnits):
    unit = 2.4188843265857e-17
    SI: dict = AtomicUnits.convert(unit, 1.0)

class Velocity(AtomicUnits):
    unit = 2.18769126364e+6
    SI: dict = AtomicUnits.convert(unit, 1.0)

class ElectricPotential(AtomicUnits):
    unit = 27.211386245988
    SI: dict = AtomicUnits.convert(unit, 1.0)