import sys
sys.stdin = open('input.txt')
sys.stdout = open('output.txt', 'w')

age = int(input())

if(age < 18):
    print("You are not an adult.")
else:
    print("You are an adult.")


if __name__ == "__main__":
    print("Hello World!")
    print(" ".isalnum())