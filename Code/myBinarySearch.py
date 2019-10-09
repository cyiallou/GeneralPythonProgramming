"""VARIOUS IMPLEMENTATIONS OF BINARY SEARCH.

TASK
-----
Given a sorted (in ascending order) integer array nums of n elements and a target value, write a function to search
for target in nums. If target exists, then return its index, otherwise return -1.

EXAMPLES
--------
Example 1:
>>> BinarySearchAlgorithms().binary_search_v1(nums=[-1, 0, 3, 5, 9, 12], target=9)
Output: 4
Explanation: 9 exists in nums and its index is 4

Example 2:
>>> BinarySearchAlgorithms().binary_search_v1(nums=[-1, 0, 3, 5, 9, 12], target=2)
Output: -1
Explanation: 2 does not exist in nums so return -1

Notes:
1) Assume that all elements in nums are unique.
2) n will be in the range [1, 10000].
3) The value of each element in nums will be in the range [-9999, 9999].
"""
import pprint
from typing import List, Callable


class BinarySearchAlgorithms:

    def __init__(self):
        pass

    @staticmethod
    def binary_search_template1(nums: List[int], target: int) -> int:
        """Perform binary search to find target in nums.

        Arguments:
            nums: List[int], sorted integer array
            target: int, target value to search for in nums

        Returns:
            idx: int, index of target in nums if target exists, otherwise -1
        """
        left, right = 0, len(nums) - 1
        while left <= right:
            mid = left + (right - left) // 2  # to avoid overflow
            if nums[mid] == target:
                return mid
            elif target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        return -1

    @staticmethod
    def binary_search_template2(nums: List[int], target: int) -> int:
        if len(nums) == 0:
            return -1

        left, right = 0, len(nums)
        while left < right:
            mid = left + (right - left) // 2  # to avoid overflow
            if nums[mid] == target:
                return mid
            elif nums[mid] < target:
                left = mid + 1
            else:
                right = mid
        # End Condition: left == right
        if left != len(nums) and nums[left] == target:
            return left
        return -1

    @staticmethod
    def binary_search_template3(nums: List[int], target: int) -> int:
        if len(nums) == 0:
            return -1

        left, right = 0, len(nums) - 1
        while left + 1 < right:
            mid = left + (right - left) // 2  # to avoid overflow
            if nums[mid] == target:
                return mid
            elif nums[mid] < target:
                left = mid
            else:
                right = mid
        # End Condition: left + 1 == right
        if nums[left] == target:
            return left
        if nums[right] == target:
            return right
        return -1

    @staticmethod
    def search_rotated_sorted_v1(nums: List[int], target: int) -> int:
        """Find target in a rotated sorted array nums.

        Suppose an array sorted in ascending order is rotated at some unknown pivot point to give nums.

        Arguments:
            nums: List[int], sorted integer array
            target: int, target value to search for in nums

        Returns:
            If target is found in the array return its index, otherwise return -1.

        Notes:
            The array elements are assumed to be unique.
            Time complexity is in the order of O(log n).
        """
        left, right = 0, len(nums) - 1
        m = (right + left) // 2
        check_values = [1, 1, 1]
        while left <= right:
            cnt = 0
            for i in [m, left, right]:
                res = nums[i] - target
                if not res:
                    return i
                else:
                    check_values[cnt] = res // abs(res)
                    cnt += 1

            if abs(sum(check_values)) == 3:
                if (nums[m] - nums[right]) > 0:  # go right
                    left, right = m + 1, right - 1
                else:  # go left
                    left, right = left + 1, m - 1
            else:
                if ((check_values[1] < 0) and (check_values[2] > 0) and (check_values[0] > 0)) \
                        or \
                        ((check_values[1] < 0) and (nums[m] - nums[right]) > 0):  # go left
                    left, right = left + 1, m - 1
                else:  # go right
                    left, right = m + 1, right - 1
            m = (right + left) // 2
        return -1

    @staticmethod
    def search_rotated_sorted_v2(nums: List[int], target: int) -> int:
        lo, hi = 0, len(nums) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if (nums[0] > target) ^ (nums[0] > nums[mid]) ^ (target > nums[mid]):
                lo = mid + 1
            else:
                hi = mid
        return lo if target in nums[lo:lo + 1] else -1


class BinarySearchAlgorithmsUnittests:
    @staticmethod
    def binary_search_sorted(func: Callable[[List[int], int], int]):
        """Unit tests for binary search algorithms on sorted arrays.

        Arguments:
            func: Callable, Must be set to binary_search_template() (1 2 or 3) from class BinarySearchAlgorithms.

        :return: None
        """
        print('Running unit tests for binary search algorithms on sorted arrays:\n')
        # {test number: (expected result, nums, target)}
        data = {1: (4, [-1, 0, 3, 5, 9, 12], 9),
                2: (-1, [-1, 0, 3, 5, 9, 12], 2),
                3: (-1, [1], 3),
                4: (0, [1], 1),
                5: (1, [1, 2], 2),
                6: (0, [2, 5], 2),
                7: (2, [-1, 0, 5], 5),
                8: (0, [-1, 0, 5], -1)}

        s = 'Test cases'
        print(f'{s}\n{"-"*len(s)}\nData structure: ', '{test number: (expected result, sorted array, target)}')
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(data)
        test_counter = 1
        for key, val in data.items():
            out = func(val[1], val[2])
            assert(out == val[0]), f'Expected {val[0]}, got {out}, for data input {key}'
            print(f'Test case {test_counter} of {len(data.items())}: Success')
            test_counter += 1

    @staticmethod
    def binary_search_rotated_sorted(func: Callable[[List[int], int], int]):
        """Unit tests for binary search algorithms on rotated sorted arrays.

        Arguments:
            func: Callable, Must be set to search_rotated_sorted_v1() or search_rotated_sorted_v2() from class
                            BinarySearchAlgorithms.
        :return: None
        """
        print('Running unit tests for binary search algorithms on rotated sorted arrays:\n')
        nums_in = [[1, 2, 3, 4, 5, 6],
                   [1, 2, 3, 4, 5, 6],
                   [8, 1, 2, 3, 4, 5, 6, 7],
                   [6, 8, 10, 12, 0, 2, 4],
                   [4, 5, 6, 7, 0, 1, 2],
                   [4, 5, 6, 7, 0, 1, 2],
                   [5, 1, 2, 3, 4]]
        target_in = [4, 2, 6, 0, 4, 5, 1]
        results = [3, 1, 6, 4, 0, 1, 1]
        # target_in = [-2 for i in range(len(target_in))]
        # results = [-1 for i in range(len(target_in))]
        test_counter = 1
        for i in range(len(target_in)):
            a = func(nums_in[i], target_in[i])
            assert (a == results[i]), f'Expected {results[i]}, got {a}. Iteration: {test_counter}'
            print(f'Test case {test_counter} of {len(target_in)}: Success')
            test_counter += 1


def main():
    """
    Runs the unit tests for the methods in class BinarySearchApps

    :return: None
    """
    A = BinarySearchAlgorithms
    B = BinarySearchAlgorithmsUnittests

    B.binary_search_sorted(func=A.binary_search_template1)
    print('\n')
    B.binary_search_sorted(func=A.binary_search_template2)
    print('\n')
    B.binary_search_sorted(func=A.binary_search_template3)
    print('\n')
    B.binary_search_rotated_sorted(func=A.search_rotated_sorted_v1)
    print('\n')
    B.binary_search_rotated_sorted(func=A.search_rotated_sorted_v2)


if __name__ == '__main__':
    main()
