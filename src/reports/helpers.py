import math


def format_number(number: float):
    if number == 0:
        return 0

    if 1e-1 <= number <= 1e6:
        return round(number, 2)
    exponent = round(math.log10(abs(number)))
    mantissa = number / (10 ** exponent)
    formatted_number = f'{mantissa:.2f}\cdot10^{{{exponent}}}'
    return formatted_number
