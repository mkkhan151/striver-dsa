from collections import Counter
import heapq
import sys
sys.stdin = open('input.txt')
sys.stdout = open('output.txt', 'w')

def reverse_words(s: str) -> str:
    """Return the sentence in reverse order."""
    result = []
    for word in s.strip().split(' '):
        if word:
            result.append(word)
    return " ".join(reversed(result))

def remove_outer_paranthesis(s: str) -> str:
    op = cl = 0
    st = 0
    result = ""
    for i in range(len(s)):
        if s[i] == '(':
            op += 1
        else:
            cl += 1
        
        if op == cl:
            op, cl = 0, 0
            result += s[st+1:i]
            st = i + 1
    return result

def largest_odd_number(num: str) -> str:
    for i in range(len(num) - 1, -1, -1):
        if int(num[i]) % 2 == 1:
            return num[:i+1]
    return ""

def longest_common_prefix(strs: list[str]) -> str:
    min_len = 201
    min_str: str | None = None
    for s in strs:
        if len(s) < min_len:
            min_len = len(s)
            min_str = s
    if min_str == "" or min_str is None:
        return ""
    for i in range(min_len, 0, -1):
        if all([min_str[:i] == s[:i] for s in strs]):
            return min_str[:i]
    return ""
    
def is_isomorphic(s: str, t: str) -> bool:
    """Returns True if two strings are isomorphic otherwise False"""
    if len(s) != len(t):
        return False

    s_to_t = {}
    t_to_s = {}

    for s_ch, t_ch in zip(s, t):
        if s_ch in s_to_t:
            if s_to_t[s_ch] != t_ch:
                return False
        else:
            s_to_t[s_ch] = t_ch

        
        if t_ch in t_to_s:
            if t_to_s[t_ch] != s_ch:
                return False
        else:
            t_to_s[t_ch] = s_ch
    return True

def rotate_string(s: str, goal: str) -> bool:
    """Returns true if s can become goal after some rotations otherwise false"""
    if len(s) != len(goal):
        return False
    if s == goal:
        return True
    for i in range(len(s) - 1):
        if s[i+1:] + s[:i+1] == goal:
            return True
    return False

def is_anagram(s: str, t: str) -> bool:
    """Returns True if s and t are anagrams otherwise false"""
    return sorted(s) == sorted(t)

def frequency_sort(s: str) -> str:
    """Returns the sorted string s based on the frequency of characters in decreasing order"""
    counter = Counter(s)
    pq = [(-freq, char) for char, freq in counter.items()]
    heapq.heapify(pq)
    result = ''
    while pq:
        freq, char = heapq.heappop(pq)
        result += char * -freq
    return result

def max_depth(s: str) -> int:
    """Returns the max depth of paranthesis in Valid Paranthesis String"""
    max_d = 0
    depth = 0
    for c in s:
        if c == '(':
            depth += 1
            max_d = max(max_d, depth)
        elif c == ')':
            depth -= 1
    return max_d

def roman_to_int(s: str) -> int:
    """Returns the roman numerals in integer form"""
    romans = {
        'I': 1,
        'V': 5,
        'X': 10,
        'L': 50,
        'C': 100,
        'D': 500,
        'M': 1000
    }
    num = 0
    for i in range(len(s) - 1, -1, -1):
        if i + 1 < len(s) and romans[s[i]] < romans[s[i+1]]:
                num -= romans[s[i]]
        else:
                num += romans[s[i]]
    return num

def my_Atoi(s: str) -> int:
    """returns int of str number"""
    i = 0
    result = 1
    # Ignore leading white spaces
    while i < len(s) and s[i] == ' ':
        i += 1

    if i >= len(s):
        return 0

    if s == '':
        return 0
    # check if signed
    if s[i] == '-':
        result = -1
        i += 1
    elif s[i] == '+':
        i += 1
    
    # read digits
    num = ''
    while i < len(s) and s[i].isdigit():
        num += s[i]
        i += 1
    
    if num == '':
        return 0
    
    result *= int(num)
    if result < -2 ** 31:
        result = -2 ** 31
    elif result > (2 ** 31 - 1):
        result = 2 ** 31 - 1
    return result

def substr_count(s: str, k: int) -> int:
    """Returns the count of substrings with k dintinct element in substring"""
    # count = 0
    # for i in range(len(s)):
    #     for j in range(i, len(s)):
    #         if len(set(s[i:j+1])) == k:
    #             count += 1
    # return count
    
    # two pointers approach
    count = 0
    low, high = 0, len(s) - 1
    while low <= high:
        if len(set(s[low:high+1])) == k: 
            count += 1
        if len(set(s[low+1: high + 1])) == k:
            count += 1
        if len(set(s[low: high])) == k:
            count += 1
        low += 1
        high -= 1
    return count

def beauty_sum(s: str) -> int:
    ans = 0
    for i in range(len(s)):
        freq = [0] * 26
        for j in range(i, len(s)):
            freq[ord(s[j]) - 97] += 1
            ans += max(freq) - min([x for x in freq if x])
    return ans

def longest_palindrome(s: str) -> str:
    """Returns the longest pallindrome substring"""
    if len(s) <= 1:
        return s
    
    def expand_from_center(left, right):
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
        return s[left + 1: right]
    
    max_str = s[0]
    for i in range(len(s) - 1):
        odd = expand_from_center(i, i)
        even = expand_from_center(i, i + 1)
        if len(odd) > len(max_str):
            max_str = odd
        if len(even) > len(max_str):
            max_str = even
    return max_str

if __name__ == '__main__':
    # print(reverse_words('a good   example'))

    # strs = ["dog","racecar","car"]
    # print(longest_common_prefix(strs))
    
    # s = "badc"
    # t = "baba"
    # print(is_isomorphic(s, t))

    # s, goal = "abcde", "abced"
    # print(rotate_string(s, goal))

    # s, t = "anagram", "nagaram"
    # print(is_anagram(s, t))

    # s = 'cccaaa'
    # print(frequency_sort(s))

    # s = '()(())((()()))'
    # print(max_depth(s))

    s = input()
    # print(roman_to_int(s))
    # print(my_Atoi(s))
    # print(substr_count(s, 1))
    # print(beauty_sum(s))
    print(longest_palindrome(s))