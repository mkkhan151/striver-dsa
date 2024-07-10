import sys
sys.stdin = open('input.txt')
sys.stdout = open('output.txt', 'w')

# An Amrstrong number is a number that is equal to the sum of its own digits each raised to the power of the number of digits.
def armstrong_number(n: int) -> bool:
    """
        Returns True if n is Armstrong number otherwise False

        Arguments:
            n -> int: A positive integer
    """
    dup_n = n
    k = len(str(n))
    sum = 0

    while n > 0:
        sum += (n % 10) ** k
        n = n // 10
    
    if sum == dup_n:
        return True

    return False

if __name__ == '__main__':
    n = int(input())
    print(armstrong_number(n))