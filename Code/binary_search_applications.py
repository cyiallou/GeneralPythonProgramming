"""
TODO: Write unit tests for guess_number
TODO: Write unit tests for find_peak_element
TODO: Improve find_peak_element to find all peaks in an array
"""
import math
import string
from typing import List


class BinarySearchApps:

    def __init__(self):
        self.pick = 1233.12374  # number to find (for guess_number functions)

    def my_real_sqrt(self, x: int) -> int:
        """
        Time complexity: O(log N).
        Space complexity: O(1).
        """
        if x < 0:
            raise ValueError('Input x cannot be negative')
        if x < 2:
            return x
        left, right = 2, x // 2
        while left <= right:
            mid = (left + right) // 2
            cond = mid ** 2
            if cond == x:
                return mid
            if cond < x:
                left = mid + 1
            else:
                right = mid - 1
        return left-1

    def my_real_sqrt_cheat(self, x: int) -> int:
        """
        Time complexity: O(1).
        Space complexity: O(1).
        """
        if x < 2:
            return x
        left = int(math.e ** (0.5 * math.log(x)))
        right = left + 1
        return left if right * right > x else right

    def my_real_sqrt_recursion(self, x: int) -> int:
        """
        Time complexity: O(log N).
        Space complexity: O(log N).
        """
        if x < 0:
            raise ValueError('Input x cannot be negative')
        if x < 2:
            return x
        left = self.my_real_sqrt_recursion(x >> 2) << 1
        right = left + 1
        return left if right * right > x else right

    def my_real_sqrt_newton(self, x: int) -> int:
        """Employ Newton's method for square root computation.

        Time complexity: O(log N).
        Space complexity: O(1).
        """
        if x < 0:
            raise ValueError('Input x cannot be negative')
        if x < 2:
            return x
        x0 = 2 ** len(str(x))  # approximate sqrt(x) as 2^n for the seed value
        x0 = x
        x1 = (x0 + x / x0) / 2
        while abs(x0 - x1) >= 1:
            x0 = x1
            x1 = (x0 + x / x0) / 2
        return int(x1)

    def guess_number(self, n: int) -> int:
        if self.pick > n:
            raise ValueError('Number to pick cannot be outside the provided range')
        left, right = 1, n
        num_picked = (right + left) / 2
        not_found = self.evaluate_guess(num_picked)
        while not_found:
            if not_found == 1:
                left = num_picked + 1
            else:
                right = num_picked - 1
            num_picked = (right + left) / 2
            not_found = self.evaluate_guess(num_picked)
        return num_picked

    def guess_number_fancy(self, n: int) -> int:
        if self.pick > n:
            raise ValueError('Number to pick cannot be outside the provided range')
        I = [0, 1, n]
        while not I[0]:
            mid = sum(I) / 2
            off = self.evaluate_guess(mid)
            I[off] = mid + off
        return I[0]

    def evaluate_guess(self, g: float) -> int:
        if self.pick != g:
            return -1 if self.pick < g else 1
        return 0

    @ staticmethod
    def find_peak_element(nums: List[int], m=2) -> int:
        """Return the index of a peak element in an array of integers.

        Arguments:
            nums: List[int], Input array
            m: int, Method to use

        Returns:
             The index of a peak element. -1 if there is no peak
        """
        if len(nums) == 0:
            return -1

        if m == 1:
            # Linear scan (complexity: O(n)):
            # check for edge cases
            if (len(nums) == 1) or (nums[0] > nums[1]):
                return 0
            if nums[-1] > nums[-2]:
                return len(nums)-1
            # check for normal cases
            for pnt in range(1, len(nums) - 1):
                if nums[pnt-1] < nums[pnt] > nums[pnt+1]:
                    return pnt
            return -1

        if m == 2:
            # Binary search (complexity: O(log n))
            # Intuition:
            #   /\
            #  /  \
            # /    \
            left, right = 0, len(nums) - 1
            while left < right:
                m = left + (right - left) // 2
                if nums[m] > nums[m + 1]:
                    right = m
                else:
                    left = m + 1
            return left

    @ staticmethod
    def minimum_in_rotated_sorted_array(nums: List[int]) -> int:
        """Uses binary search.

        Time complexity: O(log n)

        Intuition
        ---------
        Sorted array:
             /
            /
           /
          /
         /
        /
        Rotated sorted array:
          /
         /
        /
              /
             /
            /
        """
        if len(nums) == 0:
            return -1
        elif (len(nums) == 1) or (nums[0] < nums[-1]):
            return nums[0]

        left, right = 0, len(nums) - 1
        while left < right:
            m = left + (right - left) // 2
            if nums[m] < nums[m - 1]:
                return nums[m]
            elif nums[m] > nums[m + 1]:
                return nums[m + 1]
            else:
                if nums[m] > nums[left]:
                    left = m + 1
                else:
                    right = m - 1
        return nums[left]

    @staticmethod
    def minimum_in_rotated_sorted_array_shorter(nums: List[int]) -> int:
        """Simpler code with a different condition than in the normal binary search algorithm.
        """
        left, right = 0, len(nums) - 1
        while left < right:
            m = left + (right - left) // 2
            if nums[m] > nums[right]:
                left = m + 1
            else:
                right = m
        return nums[left]

    @staticmethod
    def search_range(nums: List[int], target: int) -> List[int]:
        """Find the start and end indices of the target in nums.

        TASK
        ----
        Given an array of integers (nums) sorted in ascending order, find
        the starting and ending indices of a given target value.
        If the target is not found in the array the function returns [-1, -1].

        Algorithm runtime complexity must be in the order of O(log n).

        EXAMPLES
        --------
        Example 1:
        >>> search_range(nums=[5,7,7,8,8,10], target = 8)
        Output: [3,4]

        Example 2:
        >>> search_range(nums=[5,7,7,8,8,10], target = 6)
        Output: [-1,-1]

        Arguments:
            nums: List[int], Sorted array of integers (ascending order)
            target: int, Target value

        Returns:
            out: List[int], [start index, end index] of the target in nums
        """
        if len(nums) == 0:
            return [-1, -1]

        left, right = 0, len(nums) - 1
        out = [-4, -4]  # random choice
        while left < right:
            m = left + (right - left) // 2
            if nums[m] == target:
                if nums[left] < target:
                    left = left + 1
                else:
                    out[0] = left
                if nums[right] > target:
                    right = right - 1
                else:
                    out[1] = right
            elif nums[m] < target:
                left = m + 1
            else:
                right = m
            if (out[0] != -4) and (out[1] != -4):
                return out
        if nums[left] == target:
            out[0] = left
        if nums[right] == target:
            out[1] = right
        return out if sum(out) >= 0 else [-1, -1]

    @staticmethod
    def find_closest_elements(arr: List[int], k: int, x: int) -> List[int]:
        """Find the k closest elements to x in the sorted array arr.

        EXAMPLES
        --------
        Example 1:
        >>> find_closest_elements(arr=[1,2,3,4,5], k=4, x=3)
        Output: [1,2,3,4]

        Example 2:
        >>> find_closest_elements(arr=[1,2,3,4,5], k=4, x=-1)
        Output: [1,2,3,4]

        Arguments:
            arr: List[int], Sorted array of integers
            k: int, The number of elements to return
            x: int, Target value

        Returns:
            Array of k elements sorted in ascending order (a sub-array of input arr)

        Implementation notes:
            If there is a tie, the smaller elements are always preferred.
        """
        # check input arguments
        if k < 1 or k > len(arr):
            raise ValueError('The value k must be positive and smaller than the length of the sorted array.')

        left, right = 0, len(arr) - k
        while left < right:
            mid = left + (right - left) // 2
            if x - arr[mid] > arr[mid + k] - x:
                left = mid + 1
            else:
                right = mid
        return arr[left:left + k]

    @staticmethod
    def __find_closest_elements_NOTWORKING(arr: List[int], k: int, x: int) -> List[int]:
        pass
        # take care of the edge cases
        if (len(arr) == 0) or (k > len(arr)):
            raise ValueError('k cannot be larger than the array length')
        if (len(arr) == 1) or k == len(arr):
            return arr
        if k == 0:
            return []
        left, right = 0, len(arr) - 1
        if x <= arr[left]:
            return arr[:k]
        if x >= arr[right]:
            return sorted(arr[:right - k:-1]) if k < right + 1 else arr

        lo = (x - arr[left], left)  # to hold the value and the index of the element closest to x
        fl = 1
        D = {str(i): (math.inf, math.inf) for i in range(k)}  # to hold (x - element_val, element_idx)
                                                    # of the k closest elements to x
        while left <= right:
            m = left + (right - left) // 2
            if arr[m] == x:
                fl = 0
            elif arr[m] > x:
                right = m - 1
            else:
                left = m + 1
            # find the element closest to x (favor smaller numbers)
            if abs(x - arr[m]) < lo[0]:
                lo = (abs(x - arr[m]), m)
            if fl == 0:
                break
        D['0'] = lo
        out_idx = [math.inf] * k
        out_idx[0] = lo[1]  # index
        for i in [1, 2]:  # 1: left, 2: right
            for key, val in D.items():
                if key != '0':
                    if i == 1:  # go left
                        idx = lo[1] - int(key)
                        if idx > -1:
                            D[key] = (abs(x - arr[idx]), idx)
                            out_idx[int(key)] = idx
                    else:  # go right
                        # print([arr[item] for item in out_idx])
                        idx = lo[1] + int(key)
                        if idx < len(arr):  # D[str(k-1)] != math.inf
                            for cnt, item in enumerate(D.values()):
                                if item[0] == math.inf:
                                    cond = cnt
                                    break
                                else:
                                    cond = len(out_idx) - 1
                            while cond > -1:  # < len(D):
                                if abs(x - arr[idx]) < D[str(cond)][0]:
                                    D[str(cond)] = (abs(x - arr[idx]), idx)
                                    out_idx[cond] = idx
                                    break
                                cond -= 1
        # out_idx.sort()
        return sorted([arr[item] for item in out_idx])

    @staticmethod
    def next_greatest_letter(letters: List[str], target: str, md=1) -> str:
        """Find the smallest element in the list that is larger than the given target.

        Arguments:
            letters: List[str], List of sorted characters (only lowercase)
            target: str, Target letter
            md: int, Choose solution to use

        Notes:
            Letters wrap around. For example, if target = 'z' and letters = ['a', 'b'], the answer is 'a'.
        """
        if md == 1:  # Using binary search:
            if target >= letters[-1]:
                return letters[0]

            left = 0
            right = len(letters) - 1
            while left + 1 < right:
                mid = (left + right) // 2
                if letters[mid] <= target:
                    left = mid
                else:
                    right = mid
            if letters[left] > target:
                return letters[left]
            else:
                return letters[right]

        if md == 2:  # linear scan:
            if (target < letters[0]) or (target >= letters[-1]):
                return letters[0]
            target_idx = string.ascii_lowercase.index(target)
            for i in range(1, 26):
                idx = (target_idx + i) % 26
                if not idx:
                    return letters[0]
                next_letter = string.ascii_lowercase[idx]
                if (next_letter > target) and (next_letter in letters):
                    return next_letter


class BinarySearchAppsUnitTests:
    @staticmethod
    def my_real_sqrt_unittests(c):
        if isinstance(c, BinarySearchApps):
            test_cases = [i for i in range(10000)]
            test_cases += [i**2 for i in range(10000)]
            for i in range(4):
                if i == 0:
                    try:
                        print('Running unit tests for my_real_sqrt():')
                        for i, test in enumerate(test_cases):
                            out = c.my_real_sqrt(test)
                            true_val = int(math.sqrt(test))
                            assert(out == true_val), f'For input {test}: Expected {true_val}, got {out}'
                        print('Success')
                    except AssertionError:
                        print(f'Failed for test case {i} with input: {test}')

                elif i == 1:
                    try:
                        print('Running unit tests for my_real_sqrt_cheat():')
                        for i, test in enumerate(test_cases):
                            out = c.my_real_sqrt_cheat(test)
                            true_val = int(math.sqrt(test))
                            assert(out == true_val), f'For input {test}: Expected {true_val}, got {out}'
                        print('Success')
                    except AssertionError:
                        print(f'Failed for test case {i} with input: {test}')

                elif i == 2:
                    try:
                        print('Running unit tests for my_real_sqrt_recursion():')
                        for i, test in enumerate(test_cases):
                            out = c.my_real_sqrt_recursion(test)
                            true_val = int(math.sqrt(test))
                            assert(out == true_val), f'For input {test}: Expected {true_val}, got {out}'
                        print('Success')
                    except AssertionError:
                        print(f'Failed for test case {i} with input: {test}')

                elif i == 3:
                    try:
                        print('Running unit tests for my_real_sqrt_newton():')
                        for i, test in enumerate(test_cases):
                            out = c.my_real_sqrt_newton(test)
                            true_val = int(math.sqrt(test))
                            assert(out == true_val), f'For input {test}: Expected {true_val}, got {out}'
                        print('Success')
                    except AssertionError:
                        print(f'Failed for test case {i} with input: {test}')
            return True
        else:
            raise TypeError('C is not an instance of BinarySearchApps')

    @staticmethod
    def find_min_in_rotated_sorted_array_unittests(c):
        if isinstance(c, BinarySearchApps):
            print('Running find_min_in_rotated_sorted_array_unittests():')
            test_cases = [[1, 2, 4, 5, 6, 7, 0],
                          [0, 1, 2, 4, 5, 6, 7],
                          [4, 5, 6, 7, 8, 9, 10, 11, 12, 0, 1, 2],
                          [11, 12, 0, 1, 2, 4, 5, 6, 7, 8, 9, 10],
                          [0, 1],
                          [0]]
            target_output = 0
            cnt = 0
            for test in test_cases:
                cnt += 1
                func_out = None
                try:
                    func_out = c.minimum_in_rotated_sorted_array_shorter(test)
                    assert (func_out == target_output)
                    print(f'Test case {cnt}: Passed')
                except AssertionError:
                    print(f'Test case {cnt}: Failed. Input: {test}.\n\tReturned {func_out} instead of {target_output}')

    @staticmethod
    def search_range_unittests(c):
        if isinstance(c, BinarySearchApps):
            print('Running search_range_unittests():')
            arr_in = [0, 1, 1, 1, 1, 1, 1, 1, 2, 3, 3, 3, 4, 5, 7, 7, 8, 8, 8, 8, 10, 10, 10, 10, 10, 10, 10, 10, 11]
            target = [0, 1, 2, 3, 4, 5, 7, 8, 10, 11, 9, 0, 0, 1, 1, 5]  # The last five elements are for arr_in2
            target_output = [[0, 0],
                             [1, 7],
                             [8, 8],
                             [9, 11],
                             [12, 12],
                             [13, 13],
                             [14, 15],
                             [16, 19],
                             [20, 27],
                             [28, 28],
                             [-1, -1],
                             [-1, -1],
                             [0, 0],
                             [-1, -1],
                             [0, 1],
                             [2, 2]]
            arr_in2 = [[], [0], [0], [1, 1], [1, 3, 5]]
            for i, test in enumerate(target):
                func_out = None
                try:
                    if i < 11:
                        func_out = c.search_range(nums=arr_in, target=test)
                    else:
                        func_out = c.search_range(nums=arr_in2[i-11], target=test)
                    assert (func_out == target_output[i])
                    print(f'Test case {i+1}: Passed')
                except AssertionError:
                    print(f'Test case {i+1}: Failed. Input: {test}.\n\tReturned {func_out} instead of {target_output[i]}')

    @staticmethod
    def find_closest_elements_unittests(c):
        print('Running find_closest_elements_unittests():')
        arr = [[],
               [1],
               [1],
               [1, 2, 3, 4, 5],
               [1, 2, 3, 4, 5],
               [0, 1, 1, 1, 2, 3, 6, 7, 8, 9],
               [0, 0, 1, 2, 3, 3, 4, 7, 7, 8],
               [1, 2, 5, 5, 6, 6, 7, 7, 8, 9],
               [0, 1, 2, 2, 2, 3, 6, 8, 8, 9]]
        k = [4, 2, 1, 4, 4, 9, 3, 7, 5]
        x = [3, 1, 1, 3, -1, 4, 5, 7, 9]
        target_output = [None,
                         None,
                         [1],
                         [1, 2, 3, 4],
                         [1, 2, 3, 4],
                         [0, 1, 1, 1, 2, 3, 6, 7, 8],
                         [3, 3, 4],
                         [5, 5, 6, 6, 7, 7, 8],
                         [3, 6, 8, 8, 9]]
        for i, test in enumerate(arr):
            func_out = None
            try:
                func_out = c.find_closest_elements(arr=test, k=k[i], x=x[i])
                assert(func_out == target_output[i])
                print(f'Test case {i+1}: Passed')
            except (AssertionError, ValueError):
                print(f'Test case {i+1}: Failed. Input: {test}.\n\tReturned {func_out} instead of {target_output[i]}')

    @staticmethod
    def next_greatest_letter_unittests(c):
        print('Running next_greatest_letter_unittests():')
        letters = ['c', 'f', 'j']
        target_case = ['a', 'c', 'd', 'g', 'j', 'k']
        target_output = ['c', 'f', 'f', 'j', 'c', 'c']
        for i, test in enumerate(target_case):
            func_out = None
            try:
                func_out = c.next_greatest_letter(letters=letters, target=test, md=2)
                assert(func_out == target_output[i])
                print(f'Test case {i+1}: Passed')
            except (AssertionError, ValueError):
                print(f'Test case {i+1}: Failed. Input: {test}.\n\tReturned {func_out} instead of {target_output[i]}')


def main(run_test=0):
    """Runs the unit tests for the methods in class BinarySearchApps()
    :param run_test: int, Specify the algorithm to test. Set to 0 to test all algorithms.
    :return: None
    """
    A = BinarySearchApps()

    if run_test == 1 or run_test == 0:
        BinarySearchAppsUnitTests.my_real_sqrt_unittests(A)
        print('\n')

    if run_test == 2 or run_test == 0:
        print(A.guess_number(10**24))
        print(A.guess_number_fancy(10**24))
        print('\n')

    if run_test == 3 or run_test == 0:
        BinarySearchAppsUnitTests.find_min_in_rotated_sorted_array_unittests(A)
        print('\n')

    if run_test == 4 or run_test == 0:
        BinarySearchAppsUnitTests.search_range_unittests(A)
        print('\n')

    if run_test == 5 or run_test == 0:
        BinarySearchAppsUnitTests.find_closest_elements_unittests(A)
        print('\n')

    if run_test == 6 or run_test == 0:
        BinarySearchAppsUnitTests.next_greatest_letter_unittests(A)
        print('\n')


if __name__ == '__main__':
    main(run_test=0)
