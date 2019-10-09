from typing import List


def check_conversion_to_non_decr_arr(nums: List[int]) -> bool:
    """Given an array with n integers, check if it can become non-decreasing by modifying at most 1 element.

    Non-decreasing array iff array[i] <= array[i + 1] holds for every i (1 <= i < n)
    where n belongs to [1, 10,000].

    Arguments:
        nums: List[int], Array of integers

    Returns:
        out: bool, True if array can become non-decreasing. False otherwise.


    EXAMPLES
    --------
    Example 1:
    >>> check_conversion_to_non_decr_arr(nums=[4,2,3])
    Output: True
    Explanation: You could modify the first 4 to 1 to get a non-decreasing array.

    Example 2:
    >>> check_conversion_to_non_decr_arr(nums=[4,2,1])
    Output: False
    Explanation: You can't get a non-decreasing array by modify at most one element.
    """
    out = True
    n_elements = len(nums)
    if not nums:
        raise ValueError("Input variable 'nums' is empty")
    elif n_elements > 2:
        c = 0
        for i in range(n_elements - 1):
            if (nums[i] - nums[i + 1]) > 0:
                if ((i - 1) >= 0) and ((i + 1) <= (n_elements - 2)):
                    if ((nums[i - 1] - nums[i + 1]) > 0) and ((nums[i] - nums[i + 2]) > 0):
                        c = 2
                        break
                c += 1
                if c > 1:
                    break
        if c > 1:
            out = False
    return out


def main():
    """Run a few examples.
    """
    nums_in = [[4, 2, 3, 5, 5],
               [4, 2, 1],
               [0, 0, 0],
               [0, 2, 4, 3],
               [1],
               [3, 4, 2, 3],
               [2, 3, 3, 2, 4]]
    response = [True, False, True, True, True, False, True]
    for i, test in enumerate(nums_in):
        try:
            assert(check_conversion_to_non_decr_arr(nums_in[i]) == response[i])
            print(f'Test case {i + 1}: Passed')
        except AssertionError:
            print(f'Test case {i + 1}: Failed. Input: {test}.\n')


if __name__ == '__main__':
    main()
