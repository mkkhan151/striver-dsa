import sys
sys.stdin = open('input.txt')
sys.stdout = open('output.txt', 'w')

from typing import Any


class Node:
    def __init__(self, data: Any, next = None) -> None:
        self.data = data
        self.next = next

def constructLL(arr: list[int]) -> Node:
    head = Node(arr[0])
    temp = head
    for i in range(1, len(arr)):
        temp.next = Node(arr[i])
        temp = temp.next
    return head

def get_count(head: Node | None) -> int:
    """Returns the length of list"""
    count = 0
    temp = head
    while temp:
        count += 1
        temp = temp.next
    return count

def search_key(head: Node, key: int) -> bool:
    """Returns true if key in linked list otherwise false"""
    temp = head
    while temp:
        if temp.data == key:
            return True
        temp = temp.next
    return False

def insert_at_end(head: Node | None, x: int) -> Node:
    """Returns the head of linked list after inserting x at end"""
    if not head:
        return Node(x)
    temp = head
    while temp.next:
        temp = temp.next
    temp.next = Node(x)
    return head

if __name__ == '__main__':
    # y = Node(2)
    # y.next = Node(3)
    # y.next.next = Node(4)
    # head = y
    # print(y.data)
    # while head:
    #     print(head.data, '->', end=' ')
    #     head = head.next
    # else:
    #     print('None')
    
    arr = list(map(int, input().split()))
    res = constructLL(arr)
    # while res:
    #     print(res.data, end=' ')
    #     res = res.next
    # print(get_count(res))
    # print(search_key(res, 9))
    
    