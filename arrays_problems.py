from functools import reduce
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
    print(find_len_of_longest_sub_array(arr, 10))

