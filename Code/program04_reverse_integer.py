import re
import timeit
from typing import Callable


def reverse(x: int) -> int:
    """Given a 32-bit signed integer reverse its digits.

    Returns 0 when the reversed integer overflows (i.e. outside the 32-bit signed integer range: [−2**31,  2**31 − 1]).
    """
    str_x = str(x)
    a = re.findall(r'[0-9]', str_x)
    if str_x[0] == '-':
        a += '-'
    out = int(''.join(a[::-1]))
    if (-2 ** 31) <= out <= (2 ** 31 - 1):
        return out
    else:
        return 0


def reverse_faster(x: int) -> int:
    s = (x > 0) - (x < 0)
    r = int(str(x * s)[::-1])
    return s * r * (r < 2**31)


def unittests(func: Callable[[int], int]):
    """
    Arguments:
         func: Callable, Set to either reverse or reverse_faster
    """
    test_cases = [-123450, 1534236469, -123]
    expected_result = [-54321, 0, -321]
    for i, test in enumerate(test_cases):
        func_out = None  # in case something wrong occurs in the function
        try:
            func_out = func(test_cases[i])
            assert (func_out == expected_result[i])
            print(f'Test case {i + 1}: Passed')
        except AssertionError:
            print(f'Test case {i + 1}: Failed. Input: {test}.\n')
            print(f'Expected {expected_result[i]} for input {test}, got {func_out}\n')


def reverse_wrapper():
    reverse(2**31 - 1)


def reverse_faster_wrapper():
    reverse_faster(2**31 - 1)


def main():
    unittests(func=reverse)
    unittests(func=reverse_faster)
    print(timeit.timeit(reverse_wrapper, number=100000))
    print(timeit.timeit(reverse_faster_wrapper, number=100000))


if __name__ == '__main__':
    main()