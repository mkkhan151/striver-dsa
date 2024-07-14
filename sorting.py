import sys
sys.stdin = open('input.txt')
sys.stdout = open('output.txt', 'w')

from typing import Any

def selection_sort(arr: list[Any]) -> None:
    """
    Sort the given list arr.
    """
    n = len(arr)

    for i in range(n - 1):
        min_idx = i
        for j in range(i+1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j

        # swap min_idx and i
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return

def bubble_sort(arr: list[Any]) -> None:
    """
    Sort the given list arr
    """
    n = len(arr)
    for i in range(n - 1):
        for j in range(n - i - 1):
            if arr[j] > arr[j+1]:
                # swap the adjacent elements
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return

def insertion_sort(arr: list[Any]) -> None:
    """
    Sort the given list arr.
    """
    n = len(arr)
    for i in range(1, n):
        j = i - 1
        curr_item = arr[i]
        while j >= 0 and curr_item < arr[j]:
            arr[j+1] = arr[j]
            j -= 1
        arr[j + 1] = curr_item
    return

def merge(arr: list, low: int, mid: int, high: int) -> None:
    """
    Merge the arrays (low to mid) and (mid+1 to high):
    """
    temp = [0] * ((high - low) + 1)
    left, right, k = low, mid+1, 0
    while left <= mid and right <= high:
        if arr[left] <= arr[right]:
            temp[k] = arr[left]
            left += 1
        else:
            temp[k] = arr[right]
            right += 1
        k += 1
    while left <= mid:
        temp[k] = arr[left]
        left += 1
        k += 1
    while right <= high:
        temp[k] = arr[right]
        right += 1
        k += 1
    
    # copy the merged array to its original array
    k, i = 0, low
    while i <= high:
        arr[i] = temp[k]
        i += 1
        k += 1
    return

def merge_sort(arr: list, low: int, high: int) -> None:
    """
    Sort the given list arr recursively
    """
    
    # base case
    if low < high:
        # divid the array in two parts
        mid = (low + high) // 2
        
        # Recursive calls
        # for left part (low to mid)
        merge_sort(arr, low, mid)
        # for right part (mid + 1 to high)
        merge_sort(arr, mid + 1, high)

        # now merge the both sorted array
        merge(arr, low, mid, high)

def bubble_sort_recursive(arr: list[Any], n: int) -> None:
    """
    Sort the given list arr using bubble sort recursively
    """
    # base case
    if n > 1:
        isSwaped = False
        # move the largest element to its correct position
        for i in range(n-1):
            if arr[i] > arr[i+1]:
                arr[i], arr[i+1] = arr[i+1], arr[i]
                isSwaped = True
        # Recursive call for n-1 elements
        if isSwaped:
            bubble_sort_recursive(arr, n-1)
    return

def insertion_sort_recursive(arr: list[Any], index: int, n: int) -> None:
    """
    Sort the given list arr using insertion sort recursively
    """

    # base case
    if index < n:
        # insert the current[index] element to the left sorted part of list
        curr_item = arr[index]
        j = index - 1
        while j >= 0 and curr_item < arr[j]:
            arr[j+1] = arr[j]
            j -= 1

        arr[j+1] = curr_item
        # Recursive call for increasing n+1 range
        insertion_sort_recursive(arr, index + 1, n)
    return

def partition(arr: list[Any], low: int, high: int) -> int:
    """
    Returns the correct position of pivot element
    """
    pivot = arr[low]
    left, right = low, high
    while left < right:
        while left <= high and arr[left] <= pivot:
            left += 1
        while right >= low and arr[right] > pivot:
            right -= 1
        
        if left < right:
            # swap left and right
            arr[left], arr[right] = arr[right], arr[left]
        
    arr[low] = arr[right]
    arr[right] = pivot
    return right

def quick_sort(arr: list[Any], low: int, high:int) -> None:
    """
    Sort ht e given list arr using quick sort recursively
    """
    # base case
    if low < high:
        # place the pivot element at its correct position
        partition_idx = partition(arr, low, high)
        # sort the left and right part of partion_idx
        quick_sort(arr, low, partition_idx - 1)
        quick_sort(arr, partition_idx + 1, high)
    return

if __name__ == '__main__':
    n = int(input())
    arr = list(map(int, input().strip().split()))

    print(arr)

    # selection_sort(arr)
    # bubble_sort(arr)
    # insertion_sort(arr)
    # merge_sort(arr, 0, n-1)
    # bubble_sort_recursive(arr, n)
    # insertion_sort_recursive(arr, 1, n)
    quick_sort(arr, 0, n - 1)

    print(arr)