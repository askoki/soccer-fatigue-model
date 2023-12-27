import math


def format_number(number: float):
    if number == 0:
        return 0

    if number <= 1e6:
        return round(number, 2)
    exponent = int(math.log10(abs(number)))
    mantissa = number / (10 ** exponent)
    formatted_number = f"{mantissa:.2f}\cdot10^{{{exponent}}}"
    return formatted_number
