import sys
sys.stdin = open('input.txt')
sys.stdout = open('output.txt', 'w')

from typing import Any

class Node:
    def __init__(self, data: Any, prev = None, next = None) -> None:
        self.data = data
        self.prev: Node | None = prev
        self.next: Node | None = next

def constructDLL(arr: list[int]) -> Node:
    head = Node(arr[0])
    temp = head
    for i in range(1, len(arr)):
        temp.next = Node(arr[i])
        temp.next.prev = temp
        temp = temp.next
    return head

def add_node(head: Node, p: int, x: int) -> Node:
    """insert node with value x at position p in doubly linked list"""
    temp = head
    while p > 0 and temp.next:
        temp = temp.next
        p -= 1
    node = Node(x)
    node.next = temp.next
    node.prev = temp
    if temp.next:
        temp.next = temp.next.prev = node
    else:
        temp.next = node
    return head

def delete_node(head: Node, x: int):
    """Deletes the node at position x from doubly linked list"""
    if x == 1 and head.next:
        temp = head.next
        temp.prev = None
        del head
        return temp
    temp = head
    while x > 2 and temp.next:
        temp = temp.next
        x -= 1
    if temp.next:
        d = temp.next
        temp.next = d.next
        if d.next:
            d.next.prev = temp
    return head

def reverse_DLL(head: Node | None) -> Node | None:
    """Reverse the doubly linked List"""
    if not head or not head.next:
        return head
    prev = None
    while head:
        head.prev = head.next
        head.next = prev
        prev = head
        head = head.prev
    return prev

if __name__ == '__main__':
    # y = Node(2)
    # y.next = Node(3)
    # y.next.prev = y
    # y.next.next = Node(4)
    # y.next.next.prev = y.next
    # head = y
    # print(y.data)
    # while head.next:
    #     print(head.data, '->', end=' ')
    #     head = head.next
    # while head:
    #     print(head.data, '->', end=' ')
    #     head = head.prev
    # else:
    #     print('None')

    arr = list(map(int, input().split()))
    DLL = constructDLL(arr)
    # add_node(DLL, 0, 44)
    # DLL = delete_node(DLL, 3)
    DLL = reverse_DLL(DLL)
    while DLL:
        print(DLL.data, '<->', end=' ')
        DLL = DLL.next
    else:
        print('None')
