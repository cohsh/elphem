import numpy as np

def safe_divide(a: np.ndarray | float | int, b: np.ndarray | float | int, default=float('nan')) -> np.ndarray:
    """
    Safely divides two numbers, arrays, or a combination thereof, with optional handling of division by zero.

    Args:
        a (np.ndarray | complex | float | int): Numerator, can be a number or an array.
        b (np.ndarray | complex | float | int): Denominator, can be a number or an array.
        default (complex | float | int, optional): Value to use when division by zero occurs. Default is NaN.

    Returns:
        np.ndarray: Result of the division, with division by zero handled gracefully.
    """

    dtype = np.result_type(a, b, default)
    
    a_array = np.full_like(b, a, dtype=dtype) if np.isscalar(a) else a.astype(dtype)
    
    out = np.full_like(b, fill_value=default, dtype=dtype)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.divide(a_array, b, out=out, where=b != 0)
        
    return result