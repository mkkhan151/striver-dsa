from functools import reduce
import math
from collections import defaultdict
import sys
sys.stdin = open('input.txt')
sys.stdout = open('output.txt', 'w')

# Easy Problems on Arrays

def find_largest_element(arr: list, n: int):
    """
    Returns the largest element in list
    """
    max_elem = arr[0]
    for elem in arr:
        if elem > max_elem:
            max_elem = elem
    return max_elem

def find_second_largest(arr:list, n: int) -> int:
    """
    Returns the second largest element in array/list
    """
    largest, sec_largest = arr[0], -1
    for elem in arr:
        if elem > largest:
            sec_largest = largest
            largest = elem
        elif elem > sec_largest and elem < largest:
            sec_largest = elem

    return sec_largest

def is_sorted(arr: list, n: int) -> bool:
    """
    Returns True if the array/list is sorted, False otherwise.
    """
    for i in range(n - 1):
        if arr[i] > arr[i+1]:
            return False
    return True

def is_rotated_sorted(arr: list, n: int) -> bool:
    """
    Returns True if the rotated (or not) array is sorted, otherwise False
    """
    # Brute Force: For each pos in array check if array is sorted from pos to pos - 1.
    # O(n ^ 2)
    # n = len(nums)
    # for pos in range(n):
    #     i = 0
    #     while i < n-1:
    #         index = pos + i
    #         if nums[index % n] > nums[(index + 1) % n]:
    #             break
    #         i += 1
    #     if i == n - 1:
    #         return True
    # return False

    # Optimal Solution: There will be only one position in rotated array where left/previous element is greater than right/next element
    count = 0
    n = len(arr)
    for i in range(n):
        # if current element is greater than next element, increment count
        if arr[i] > arr[(i + 1) % n]: # for i = n - 1, next index is 0 (n - 1 + 1 % n)
            count += 1
    # if count > 1, arr is not sorted and return False, otherwise return True
    return count <= 1
    
def remove_duplicates(nums: list[int]) -> int:
        unique_ptr, curr_ptr = 0, 1
        while curr_ptr < len(nums):
            if nums[curr_ptr] != nums[unique_ptr]:
                unique_ptr += 1
                nums[unique_ptr] = nums[curr_ptr]
            curr_ptr += 1
        return unique_ptr + 1

def left_rotate_by_one(arr: list[int]) -> None:
    """
    Rotates the array to left by one place
    """
    temp = arr[0]
    for i in range(1, len(arr)):
        arr[i-1] = arr[i]
    arr[-1] = temp # place temp at last position

def left_rotate_by_k(arr: list[int], k: int) -> None:
    """
    Rotates the array to left by k places
    """
    # Solution 1: Rotate the array k times by one place. TC -> O(n * k), SC -> O(1) extra
    # while k:
    #     k -= 1
    #     left_rotate_by_one(arr)

    # Solution 2: Store first k elements in a temp array and shift next (n - k) elements to left by k places
    # TC -> O(n + k), SC -> O(k) extra
    # n = len(arr)
    # temp = []
    # for i in range(k):
    #     temp.append(arr[i])
    
    # shift n - k elements to left by k places
    # for i in range(k, n):
    #     arr[i-k] = arr[i]
    
    # Append temp k elements to the right of n - k elements after shifting to the left
    # for i in range(n - k, n):
    #     arr[i] = temp[i - (n - k)]

    # Solution 3: (Optimal) Reverse first k elements and then (n - k) elements and then the whole array.
    # TC -> O(2n) , SC -> O(1)

def move_zeros_to_end(arr: list[int], n: int) -> None:
    """
    Moves all the zeros in array to the end of the array.
    """
    zero_ptr = 0
    while zero_ptr < n and arr[zero_ptr] != 0:
        zero_ptr += 1
    
    curr_ptr = zero_ptr + 1
    while curr_ptr < n:
        if arr[curr_ptr] != 0:
            # swap zero and non-zero element
            arr[zero_ptr], arr[curr_ptr] = arr[curr_ptr], arr[zero_ptr]
            zero_ptr += 1
        curr_ptr += 1
    return

def find_union(arr1: list[int], arr2: list[int]) -> list[int]:
    """
    Returns the union of two sorted arrays
    """
    left, right = 0, 0
    union = []
    n = len(arr1)
    m = len(arr2)
    while left < n and right < m:
        if arr1[left] <= arr2[right]:
            if len(union) == 0 or union[-1] != arr1[left]:
                union.append(arr1[left])
            left += 1
        else:
            if len(union) == 0 or union[-1] != arr2[right]:
                union.append(arr2[right])
            right += 1
    while left < n:
        if union[-1] != arr1[left]:
            union.append(arr1[left])
        left += 1
    while right < n:
        if union[-1] != arr2[right]:
            union.append(arr2[right])
        right += 1
    return union

def find_intersection(arr1: list[int], arr2: list[int]) -> list[int]:
    """
    Returns the intersection of two sorted arrays
    """
    intersection = []
    left, right = 0, 0
    n, m = len(arr1), len(arr2)
    while left < n and right < m:
        if arr1[left] < arr2[right]:
            left += 1
        elif arr1[left] > arr2[right]:
            right += 1
        else:
            intersection.append(arr2[right])
            left += 1
            right += 1
    return intersection

def find_missing_number(arr: list[int], n: int) -> int:
    """
    Returns the missing number in the array from 1 to n
    """
    # Brute force solution. TC -> O(n ^ 2)
    # for i in range(1, n+1):
    #     if i not in arr:
    #         return i
    # return -1

    # Better Solution. TC -> O(2n), SC -> O(n+1)
    # temp = [False] * (n+1)
    # for i in arr:
    #     temp[i] = True

    # for i in range(1, n+1):
    #     if not temp[i]:
    #         return i
    # return -1

    # Optimal Solution 1. TC -> O(n)
    # return (n * (n + 1)) // 2 - sum(arr)

    # Optimal Solution 2. TC -> O(n)
    xor1, xor2 = 0, 0
    for i in range(n-1):
        xor2 = xor2 ^ arr[i]
        xor1 = xor1 ^ (i + 1)
    xor1 = xor1 ^ n
    return xor1 ^ xor2

def find_max_consecutive_ones(arr: list[int]) -> int:
    """
    Returns the max number of consecutive ones in the array.
    """
    max_ones = 0
    count = 0
    for num in arr:
        if num == 1:
            count += 1
        else:
            max_ones = max(count, max_ones)
            count = 0
    return max(max_ones, count)

def find_single_number(nums: list[int]) -> int:
    """
    Returns the number that appears only once in the array, all other numbers appear twice.
    """
    # single = 0
    # for num in nums:
    #     single = single ^ num
    # return single
    return reduce(lambda x, y: x ^ y, nums)

def find_len_of_longest_sub_array(arr: list[int], k: int) -> int:
    """
    Returns the length of longest subarray which sum is equal to k.
    """
    # Solution 1: Generate all sub arrays using 3 loops and find the longest one. TC -> O(n ^ 3)
    # Solution 2: Generate all sub arrays using two loops and sum along with and find the longest one. TC -> O(n ^ 3)
    # Solution 3: Presum map
    # pre_sum = {}
    # curr_sum = 0
    # max_len = 0
    # for i in range(len(arr)):
    #     curr_sum += arr[i]

    #     if curr_sum == k:
    #         max_len = max(max_len, i + 1)
        
    #     rem = curr_sum - k
    #     if rem in pre_sum:
    #         max_len = max(max_len,i - pre_sum[rem])

    #     if curr_sum not in pre_sum:
    #         pre_sum[curr_sum] = i
    # return max_len

    # Solution 4: Two Pointers
    left, right = 0, 0
    max_len = 0
    curr_sum = arr[0]
    while right < len(arr):
        while left <= right and curr_sum > k:
            curr_sum -= arr[left]
            left += 1
        
        if curr_sum == k:
            max_len = max(max_len, right - left + 1)
        right += 1
        if right < len(arr): curr_sum += arr[right]
    return max_len

# Medium problems on arrays

def two_sum(nums: list[int], target: int) -> list[int]:
    """
    Returns the indeces of pair which sum is equal to target.
    """
    # Brute force solution: TC -> O(n ^2)
    # for i in range(len(nums) - 1):
    #         for j in range(i+1, len(nums)):
    #             if nums[i] + nums[j] == target:
    #                 return [i, j]
    # return [-1, -1] # in case there is no pair with sum equal to target
    
    # Better Solution using Hashing: TC -> O(n)
    visited = {}
    for i in range(len(nums)):
        sec_num = target - nums[i]

        if sec_num in visited:
            return [visited[sec_num], i]
        
        if nums[i] not in visited:
            visited[nums[i]] = i
        
    return [-1, -1]

def sort_colors(nums: list[int]) -> None:
    """
    Sort the given array of colors represented with 0, 1, and 2
    """
    low, mid, high = 0, 0, len(nums)-1
    while mid <= high:
        if nums[mid] == 0:
            nums[low], nums[mid] = nums[mid], nums[low]
            mid += 1
            low += 1
        elif nums[mid] == 2:
            nums[high], nums[mid] = nums[mid], nums[high]
            high -= 1
        else:
            mid += 1

def majority_element(nums: list[int]) -> int | None:
    """
    Returns the majority element in the array.
    """
    count = 0
    element = None

    for num in nums:
        if count == 0:
            element = num
            count = 1
        elif element == num:
            count += 1
        else:
            count -= 1
    return element

def max_sub_array_sum(nums: list[int]) -> int:
    """
    Returns maximum subarray sum
    """
    max_sum = 0
    curr_sum = 0
    for num in nums:
        curr_sum += num
        max_sum = max(max_sum, curr_sum)
        curr_sum = max(0, curr_sum)
    
    return max_sum

def max_profit(prices: list[int]) -> int:
    """
    Returns the max profit from the given prices of stock where each index represents the ith stock price.
    We buy at lowest price and sell at largest price in future to maximize the profit.
    """
    max_profit = 0
    buy = prices[0]
    for i in range(1, len(prices)):
            # calculate the current profit and update maximum
            max_profit = max(max_profit, prices[i] - buy)

            if prices[i] < buy:
                # Buy today if current prices is less than the bought price
                buy = prices[i]
    return max_profit

def rearrange_array(nums: list[int]) -> list[int]:
    """
    Returns the rearranged array of alternative signed numbers e.g: [1, -2, 3, -7]
    """
    result = [0] * len(nums)
    pos_ptr, neg_ptr = 0, 1

    for num in nums:
        if num > 0:
            result[pos_ptr] = num
            pos_ptr += 2
        else:
            result[neg_ptr] = num
            neg_ptr += 2
    return result

def next_permutation(nums: list[int]) -> None:
    """
    Arrange the given array of nums in next permutation in all sorted permutations
    """
    i = len(nums) - 2
    # find the break point of increasing order from end
    while i >= 0 and nums[i] >= nums[i+1]:
        i -= 1
    if i == -1:
        nums.reverse()
        return
    
    j = len(nums) - 1
    # find the number greater than nums[i] but the smallest after i index
    while j >= 0 and nums[j] <= nums[i]:
        j -= 1
    
    #swap i and j index elements
    nums[i], nums[j] = nums[j], nums[i]

    # reverse i+1 to end elements
    nums[i+1:] = nums[i+1:][::-1]

def find_leaders(arr: list[int]) -> list[int]:
    """
    Returns the list of leaders in the given array.
    An element of the array is considered a leader if it is greater than all the elements on its right side or if it is equal to the maximum element on its right side. The rightmost element is always a leader.
    """
    # Brute force solution, TC -> O(n ^ 2)
    # leaders = []
    # for i in range(len(arr)):
    #     is_leader = True
    #     for j in range(i+1, len(arr)):
    #         if arr[i] < arr[j]:
    #             is_leader = False
    #             break
    #     if is_leader:
    #         leaders.append(arr[i])
    # return leaders

    # Optimal Solution: TC -> O(n)
    leaders = []
    curr_max = arr[-1] # last element is always the leader
    i = len(arr) - 1
    while i >= 0:
        if arr[i] >= curr_max:
            curr_max = arr[i]
            leaders.append(curr_max)
        i -= 1
    leaders.reverse()
    return leaders

def longest_consecutive(nums: list[int]) -> int:
    """
    Return the length of longest consecutive sequence in the array
    """
    if len(nums) == 0 or len(nums) == 1:
        return len(nums)
    longest = 1
    # count = 1
    # nums.sort()
    # for i in range(1, len(nums)):
    #     if nums[i] == nums[i-1] + 1:
    #         count += 1
    #     elif nums[i] != nums[i - 1]:
    #         count = 1
    #     longest = max(longest, count)
    # return longest
    s = set(nums)
    for elem in s:
        if elem - 1 not in s:
            count = 1
            curr_elem = elem + 1
            while curr_elem in s:
                count += 1
                curr_elem += 1
            longest = max(longest, count)
    return longest

def set_zeros(matrix: list[list[int]]) -> None:
    """
    Set the corresponding column and row of the matrix cell to zero if that cell is zero
    """
    m = len(matrix)
    n = len(matrix[0])
    col0 = 1
    for i in range(m):
        for j in range(n):
            if matrix[i][j] == 0:
                # mark i-th row
                matrix[i][0] = 0

                # mark j-th column
                if j != 0:
                    matrix[0][j] = 0
                else:
                    col0 = 0

    for i in range(1, m):
        for j in range(1, n):
            if matrix[i][j] != 0:
                if matrix[i][0] == 0 or matrix[0][j] == 0:
                    matrix[i][j] = 0
    if matrix[0][0] == 0:
        for j in range(n):
            matrix[0][j] = 0
    if col0 == 0:
        for i in range(m):
            matrix[i][0] = 0

def rotate_image(matrix: list[list[int]]) -> None:
    """
    Rotates the (n x n) image by 90 degree clockwise
    """
    # Brute force: TC, SC -> O(n ^ 2)
    # n = len(matrix)
    # ans = [[0 for _ in range(n)] for _ in range(n)]
    # for i in range(n):
    #     for j in range(n):
    #         ans[j][n - 1 - i] = matrix[i][j]
    # print(ans)

    # Optimal Solution: TC -> O(n ^ 2), SC -> O(1)
    # first take transpose of matrix
    n = len(matrix)
    for i in range(0, n - 1):
        for j in range(i + 1, n):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
    # now reverse each row in matrix
    for i in range(n):
        matrix[i].reverse()

def spiral_order(matrix: list[list[int]]) -> list[int]:
    """
    Returns the spiral order of the given (m x n) matrix in 1-D resultant list
    """
    m, n = len(matrix), len(matrix[0])
    left, top, right, bottom = 0, 0, n - 1, m - 1
    spiral_result = []

    while top <= bottom and left <= right:
        for i in range(left, right + 1):
            spiral_result.append(matrix[top][i])
        top += 1

        for i in range(top, bottom + 1):
            spiral_result.append(matrix[i][right])
        right -= 1

        if top <= bottom:
            for i in range(right, left - 1, -1):
                spiral_result.append(matrix[bottom][i])
            bottom -= 1
        
        if left <= right:
            for i in range(bottom, top - 1, -1):
                spiral_result.append(matrix[i][left])
            left += 1
    return spiral_result

    # while total_elements > 0:
    #     # first row
    #     i, j = top + 1, left + 1
    #     while j < right:
    #         spiral_result.append(matrix[i][j])
    #         total_elements -= 1
    #         j += 1
    #     top += 1

    #     # last column
    #     i, j = top + 1, right - 1
    #     while i < bottom:
    #         spiral_result.append(matrix[i][j])
    #         total_elements -= 1
    #         i += 1
    #     right -= 1

    #     # last row
    #     i, j = bottom - 1, right - 1
    #     while j > left:
    #         spiral_result.append(matrix[i][j])
    #         total_elements -= 1
    #         j -= 1
    #     bottom -= 1

    #     # first column
    #     i, j = bottom - 1, left + 1
    #     while i > top:
    #         spiral_result.append(matrix[i][j])
    #         total_elements -= 1
    #         i -= 1
    #     left += 1
    # return spiral_result

def subarray_sum_count(nums: list[int], k: int) -> int:
    """
    Return the count of subarrays of sum k
    """
    n = len(nums)
    pre_sum_map = defaultdict(int)

    pre_sum_map[0] = 1 # set zero in map
    pre_sum = 0
    count = 0
    for num in nums:
        # add current element to prefix sum
        pre_sum += num

        # Calculate prefix_sum - k
        remainder = pre_sum - k

        # add the count of remainder subarrays to count
        count += pre_sum_map[remainder]

        # update the count of prefix sum in map
        pre_sum_map[remainder] += 1
    return count


if __name__ == '__main__':
    n = int(input())
    arr = list(map(int, input().strip().split()))
    # arr1 = list(map(int, input().strip().split()))
    # arr2 = list(map(int, input().strip().split()))
    # k = int(input())

    # print(find_largest_element(arr, n))
    # print(find_second_largest(arr, n))
    # print(is_sorted(arr, n))
    # print(remove_duplicates(arr))
    # print(arr)
    # left_rotate_by_one(arr)
    # left_rotate_by_k(arr, k)
    # move_zeros_to_end(arr, n)
    # print(arr)

    # print(find_union(arr1, arr2))
    # print(find_intersection(arr1, arr2))

    # print(find_missing_number(arr, n))
    # print(find_max_consecutive_ones(arr))
    # print(find_single_number(arr))
    # print(find_len_of_longest_sub_array(arr, 10))

    # sort_colors(arr)
    # print(arr)
    # print(majority_element(arr))
    # print(max_sub_array_sum(arr))
    # print(max_profit(arr))
    # print(rearrange_array(arr))
    # next_permutation(arr)
    # print(arr)
    # print(find_leaders(arr))
    # print(longest_consecutive(arr))

    # arr1 = [[1, 2, 3],[4, 5, 6],[7, 8, 9]]
    # rotate_image(arr1)
    # print(arr1)

    # print(spiral_order(arr1))

    print(subarray_sum_count(arr, 0))