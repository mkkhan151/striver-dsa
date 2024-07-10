import sys
sys.stdin = open('input.txt')
sys.stdout = open('output.txt', 'w')

def printNos(N: int) -> None:
    """
    Prints numbers from 1 to N using recursion

    Arguments:
        N: int -> A number greater than 1
    """
    if N < 1:
        return
    printNos(N-1)
    print(N, end=" ")
    return

def print_name_n_times(name: str, N: int) -> None:
    """
    prints name N times using recursion

    Arguments:
        name: str -> string to be print N times
        N: int -> number to print name N times
    """
    if N <= 0:
        return
    print(name)
    print_name_n_times(name, N - 1)
    return

def printNosReverse(N: int) -> None:
    """
    Prints numbers from 1 to N in reverse order (N to 1) using recursion

    Arguments:
        N: int -> A number greater than 1
    """
    if N < 1:
        return
    print(N, end=" ") # print the current N
    printNosReverse(N-1) # recursive call
    return

def sum_first_n(n: int) -> int:
    """
    returns the sum of first n numbers (1 to n) using recursion

    Arguments:
        n: int -> first n numbers
    """
    # return int((n * (n + 1)) / 2) # O(1) time and space complexity

    if n <= 0:
        return 0
    return n + sum_first_n(n - 1) # O(n) time and space complexity

def factorial(n: int) -> int:
    """
    returns factorial of n

    Arguments:
        n: int -> number which factorial is to be returned

    returns:
        factorial of n:
    """
    if n == 0 or n == 1:
        return 1
    return n * factorial(n - 1)

def reverse_array(arr: list[int], st: int, end: int) -> None:
    """
    reverse the array arr of length end + 1
    """
    if st < end:
        # swap st and end position values
        arr[end], arr[st] = arr[st], arr[end]

        # recursive call for next swaps
        reverse_array(arr, st + 1, end - 1)
    return

def is_palindrome(s: str, st: int = 0) -> bool:
    """
    Returns True if s is a palindrome otherwise False
    """
    if st >= len(s) // 2:
        return True
    if s[st] != s[len(s)-1-st]:
        return False
    
    return is_palindrome(s, st + 1)

def fibonacci(n: int) -> int:
    """
    Return nth fibonacci number
    """
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

if __name__ == '__main__':
    N = int(input())
    # printNos(N) 
    # print_name_n_times("Muhammad Kamran", N)
    # printNosReverse(N)
    # print(sum_first_n(N))
    # print(factorial(N))
    
    # arr = list(map(int, input().split()))
    # print(arr)
    # reverse_array(arr, 0, N-1)
    # print(arr)

    # print(is_palindrome("11211"))

    print(fibonacci(N))