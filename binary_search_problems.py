import math
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

def square_root(n: int) -> int:
    """Returns the square root of number n"""
    low, high = 1, n
    while low <= high:
        mid = (low + high) // 2
        val = mid * mid
        if val <= n:
            low = mid + 1
        else:
            high = mid - 1
    return high

def Nth_root(n: int, m: int) -> int:
    """Return nth root of m"""
    low, high = 1, m
    while low <= high:
        mid = (low + high) // 2

        val = mid ** n
        if val <= m:
            low = mid + 1
        else:
            high = mid - 1
    return high if high ** n == m else -1

def min_eating_speed(piles: list[int], h: int) -> int:
    """
    Returns min speed of eating bananas in h hours.
    """
    def calculate_hours(piles: list[int], k: int) -> int:
        """Returns the total hourse to eat bananas at k speed"""
        tot_hours = 0
        for pile in piles:
            tot_hours += math.ceil(pile / k)
        return tot_hours
    low, high = 1, max(piles)
    ans = high
    while low <= high:
        mid = (low + high) // 2
        if calculate_hours(piles, mid) <= h:
            ans = min(ans, mid)
            high = mid - 1
        else:
            low = mid + 1
    return ans

def min_days(bloom_day: list[int], m: int, k: int) -> int:
    """Returns minimum number of days to make m bouquets with k flowers"""
    if m * k > len(bloom_day):
        return -1
    def calculate_bouquets(bloom_day: list[int], curr_day: int, k: int) -> int:
        """Returns the number of bouquets at current day"""
        bouquets = 0
        count = 0
        for day in bloom_day:
            if curr_day >= day:
                count += 1
            else:
                bouquets += count // k
                count = 0
        bouquets += count // k
        return bouquets
    low, high = min(bloom_day), max(bloom_day)
    ans = high
    while low <= high:
        mid = (low + high) // 2
        if calculate_bouquets(bloom_day, mid, k) >= m:
            ans = min(ans, mid)
            high = mid - 1
        else:
            low = mid + 1
    return ans

def smallest_divisor(nums: list[int], threshold: int) -> int:
    """Returns the smallest divisor of the array"""
    def possible(nums: list[int], div: int) -> int:
        s = 0
        for num in nums:
            s += math.ceil(num / div)
        return s
    low, high = 1, max(nums)
    ans = high
    while low <= high:
        mid = (low + high) // 2
        if possible(nums, mid) <= threshold:
            ans = min(ans, mid)
            high = mid - 1
        else:
            low = mid + 1
    return ans

def least_weight_capacity(weights: list[int], days: int) -> int:
    """Returns the least capacity of conveyor belt to ship weights within given days"""
    def find_days(weights: list[int], cap: int) -> int:
        """Returns number of days to ship weights with current capacity cap"""
        days = 1
        load = 0
        for weight in weights:
            if load + weight > cap:
                days += 1
                load = weight
            else:
                load += weight
        return days
    low, high = max(weights), sum(weights)
    while low <= high:
        mid = (low + high) // 2
        if find_days(weights, mid) <= days:
            high = mid - 1
        else:
            low = mid + 1
    return low

def find_kth_positive(arr: list[int], k: int) -> int:
    """Returns the kth missing positive number in strictly increasing array"""
    low, high = 0, len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        missing = arr[mid] - (mid + 1)
        if missing < k:
            low = mid + 1
        else:
            high = mid - 1
    return k + high + 1

def findPages(arr, n, m):
    # book allocation impossible
    if m > n:
        return -1
    
    def countStudents(arr, pages):
        n = len(arr)  # size of array
        students = 1
        pagesStudent = 0
        for i in range(n):
            if pagesStudent + arr[i] <= pages:
                # add pages to current student
                pagesStudent += arr[i]
            else:
                # add pages to next student
                students += 1
                pagesStudent = arr[i]
        return students

    low = max(arr)
    high = sum(arr)
    while low <= high:
        mid = (low + high) // 2
        students = countStudents(arr, mid)
        if students > m:
            low = mid + 1
        else:
            high = mid - 1
    return low

def split_array(nums: list[int], k: int) -> int:
    """Return the minimized largest sum of the k splits."""
    def count_partitions(nums: list[int], largest_sum: int) -> int:
        """Return the number of splits/partitions with largest sum of a split"""
        curr_sum = 0
        k = 1
        for num in nums:
            if curr_sum + num > largest_sum:
                k += 1
                curr_sum = num
            else:
                curr_sum += num
        return k
    
    low, high = max(nums), sum(nums)
    while low <= high:
        mid = (low + high) // 2
        if count_partitions(nums, mid) > k:
            low = mid + 1
        else:
            high = mid - 1
    return low

def median_of_sorted_arrays(nums1: list[int], nums2: list[int]):
    """Return the median of two sorted arrays after merging them."""
    n1, n2 = len(nums1), len(nums2)
    if n1 > n2:
        return median_of_sorted_arrays(nums2, nums1)
    low, high = 0, n1
    n = n1 + n2
    left = (n + 1) // 2
    while low <= high:
        mid1 = (low + high) // 2
        mid2 = left - mid1
        # calculate l1, l2, r1, and r2;
        l1, l2, r1, r2 = float('-inf'), float('-inf'), float('inf'), float('inf')
        if mid1 < n1:
            r1 = nums1[mid1]
        if mid2 < n2:
            r2 = nums2[mid2]
        if mid1 - 1 >= 0:
            l1 = nums1[mid1 - 1]
        if mid2 - 1 >= 0:
            l2 = nums2[mid2 - 1]

        if l1 <= r2 and l2 <= r1:
            if n % 2 == 1:
                return max(l1, l2)
            else:
                return (max(l1, l2) + min(r1, r2)) / 2
        elif l1 > r2:
            high = mid1 - 1
        else:
            low = mid1 + 1
    return 0

def find_kth_element(arr1: list[int], arr2: list[int], k: int) -> int:
    n1, n2 = len(arr1), len(arr2)
    if n1 > n2:
        return find_kth_element(arr2, arr1, k)
    low, high = max(0, k - n2), min(n1, k)
    n = n1 + n2
    left = k
    while low <= high:
        mid1 = (low + high) // 2
        mid2 = left - mid1
        # calculate l1, l2, r1, and r2;
        l1, l2, r1, r2 = float('-inf'), float('-inf'), float('inf'), float('inf')
        if mid1 < n1:
            r1 = arr1[mid1]
        if mid2 < n2:
            r2 = arr2[mid2]
        if mid1 - 1 >= 0:
            l1 = arr1[mid1 - 1]
        if mid2 - 1 >= 0:
            l2 = arr2[mid2 - 1]

        if l1 <= r2 and l2 <= r1:
            return max(l1, l2)
        elif l1 > r2:
            high = mid1 - 1
        else:
            low = mid1 + 1
    return 0

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
    # print(find_single_nonduplicate(arr))
    # print(square_root(28))
    # print(Nth_root(3, 9))
    # print(min_eating_speed(arr, n))
    # print(min_days(arr, n, 1))
    # print(smallest_divisor(arr, n))
    # print(least_weight_capacity(arr, n))
    # print(find_kth_positive(arr, n))
    print(split_array(arr, n))