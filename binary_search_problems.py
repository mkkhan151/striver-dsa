import sys
sys.stdin = open('input.txt')
sys.stdout = open('output.txt', 'w')

def binary_search(nums: list[int], target: int) -> int:
    """
    Returns the index of target element in array if found otherwise -1.
    """
    # Iterative solution
    # low, high = 0, len(nums) - 1
    # while low <= high:
    #     mid = (low + high) // 2
    #     if nums[mid] == target:
    #         return mid
    #     elif nums[mid] > target:
    #         high = mid - 1
    #     else:
    #         low = mid + 1
    # return -1

    # Recursive Solution
    def search(nums: list[int], target: int, low: int, high: int) -> int:
        if low <= high:
            mid = (low + high) // 2
            if nums[mid] == target:
                return mid
            elif nums[mid] > target:
                return search(nums, target, low, mid - 1)
            else:
                return search(nums, target, mid + 1, high)
        return -1
    return search(nums, target, 0, len(nums) - 1)

def find_floor(nums: list[int], target: int) -> int:
    """
    Returns the index of element that is less or equal to target otherwise -1.
    """
    floor_idx = -1
    low, high = 0, len(nums) - 1
    while low <= high:
        mid = (low + high) // 2
        if target >= nums[mid]:
            floor_idx = mid
            low = mid + 1
        else:
            high = mid - 1
    return floor_idx

def lower_bound(nums: list[int], target: int) -> int:
    """Returns theindex of lower bound of target in array otherwise size of array"""
    low, high = 0, len(nums) - 1
    ans = high + 1
    while low <= high:
        mid = (low + high) // 2
        # maybe an answer
        if nums[mid] >= target:
            ans = mid
            # look for smaller index on the left
            high = mid - 1
        else:
            low = mid + 1 # look on the right
    return ans

def upper_bound(nums: list[int], target: int) -> int:
    """Returns theindex of lower bound of target in array otherwise size of array"""
    low, high = 0, len(nums) - 1
    ans = high + 1
    while low <= high:
        mid = (low + high) // 2
        # maybe an answer
        if nums[mid] > target:
            ans = mid
            # look for smaller index on the left
            high = mid - 1
        else:
            low = mid + 1 # look on the right
    return ans

def search_insert(nums: list[int], target: int) -> int:
    """
    Returns the inserting position of target in a sorted array
    """
    low, high = 0, len(nums) - 1
    ans = len(nums)
    while low <= high:
        mid = (low + high) // 2
        if nums[mid] >= target:
            ans = mid
            high = mid - 1
        else:
            low = mid + 1
    return ans

def first_and_last_index(nums: list[int], target: int) -> list[int]:
    """
    Returns the first and last index of the target in array if found otherwise [-1, -1]
    """
    if len(nums) == 0:
        return [-1, -1]
    # find first occurance
    first = -1
    low, high = 0, len(nums) - 1
    while low <= high:
        mid = (low + high) // 2
        if nums[mid] == target:
            first = mid
            high = mid - 1
        elif nums[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    # find the last occurance
    last = -1
    low, high = 0, len(nums) - 1
    while low <= high:
        mid = (low + high) // 2
        if nums[mid] == target:
            last = mid
            low = mid + 1
        elif nums[mid] > target:
            high = mid - 1
        else:
            low = mid + 1
    return [first, last]

def find_count(nums: list[int], target: int) -> int:
    """
    Returns the count of target in sorted array
    """
    result = first_and_last_index(nums, target)
    if result[0] == -1:
        return 0
    return result[1] - result[0] + 1

def search_rotated_sorted_array(nums: list[int], target: int) -> int:
    """
    Returns the position of target element in rotated sorted array if found otherwise -1.
    """
    low, high = 0, len(nums) - 1
    while low <= high:
        mid = (low + high) // 2
        if target == nums[mid]:
            return mid
        # identify the sorted half
        elif nums[low] <= nums[mid]:
            if nums[low] <= target and target <= nums[mid]:
                high = mid - 1
            else:
                low = mid + 1
        elif nums[mid] <= target and target <= nums[high]:
            low = mid + 1
        else:
            high = mid - 1

    return -1
            
def find_min(nums: list[int]) -> int:
    """
    Returns the minimum element in rotated sorted array.
    """
    ans = sys.maxsize
    low, high = 0, len(nums) - 1
    while low <= high:
        mid = (low + high) // 2
        if nums[low] <= nums[mid]:
            ans = min(ans, nums[low])
            low = mid + 1
        else:
            ans = min(ans, nums[mid])
            high = mid - 1
    return ans

def find_kth_rotation(arr: list[int]) -> int:
    """Returns the count of rotations in array."""
    low, high = 0, len(arr) - 1
    index, min_value = -1, sys.maxsize
    while low <= high:
        mid = (low + high) // 2
        if arr[low] <= arr[mid]:
            if arr[low] < min_value:
                min_value = arr[low]
                index = low
            low = mid + 1
        else:
            if arr[mid] < min_value:
                index = mid
                min_value = arr[mid]
            high = mid - 1
    return index

def find_single_nonduplicate(nums: list[int]) -> int:
    """Returns the element that appears only once in array."""
    low, high = 0, len(nums) - 1
    if high == 0:
        return nums[0]
    if nums[0] != nums[1]:
        return nums[0]
    if nums[high] != nums[high - 1]:
        return nums[high]
    while low <= high:
        mid = (low + high) // 2
        # Check if the mid point appears only once
        if nums[mid - 1] != nums[mid] and nums[mid] != nums[mid + 1]:
            return nums[mid]
        elif nums[mid - 1] == nums[mid]:
            if (high - mid) % 2 == 0:
                high = mid - 2
            else:
                low = mid + 1
        else:
            if (mid - low) % 2 == 0:
                low = mid + 2
            else:
                high = mid - 1
    return -1

def find_peak_element(nums: list[int]) -> int:
    """
    Returns any peak element in the array.
    """
    low, high = 1, len(nums) - 1
    if high == 0:
        return 0
    if nums[0] > nums[1]:
        return 0
    if nums[high] > nums[high - 1]:
        return high
    high -= 1
    while low <= high:
        mid = (low + high) // 2
        # if mid is peak
        if nums[mid - 1] < nums[mid] and nums[mid] > nums[mid + 1]:
            return mid
        # if we are in the left
        if nums[mid] > nums[mid - 1]:
            low = mid + 1
        else:
            high = mid - 1
    return -1

if __name__ == '__main__':
    n = int(input())
    arr = list(map(int, input().strip().split()))

    # print(binary_search(arr, n))
    # print(find_floor(arr, n))
    # print(upper_bound(arr, n))
    # print(search_insert(arr, n))
    # print(first_and_last_index(arr, n))
    # print(find_count(arr, n))
    # print(search_rotated_sorted_array(arr, n))
    # print(find_min(arr))
    # print(find_kth_rotation(arr))
    print(find_single_nonduplicate(arr))
    