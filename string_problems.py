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

if __name__ == '__main__':
    print(reverse_words('a good   example'))