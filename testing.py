"""
In this file, I write the algorithm to test if it is working before finalizing that
algorithm for a problem solution
"""

def lowerBound(arr: list[int], n: int, x: int) -> int:
    low = 0
    high = n - 1
    ans = n

    while low <= high:
        mid = (low + high) // 2
        # maybe an answer
        if arr[mid] >= x:
            ans = mid
            # look for smaller index on the left
            high = mid - 1
        else:
            low = mid + 1  # look on the right

    return ans

if __name__ == "__main__":
    arr = [1, 2, 8, 10, 11, 12, 19]
    n = 7
    x = 5
    ind = lowerBound(arr, n, x)
    print("The lower bound is the index:", ind)