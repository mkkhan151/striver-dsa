import sys
sys.stdin = open('input.txt')
sys.stdout = open('output.txt', 'w')

from typing import Any, Optional

class ListNode:
    def __init__(self, val=0, next=None) -> None:
        self.val = val
        self.next = next

def construct_LL(arr: list[Any]) -> ListNode | None:
    """Construct linked list from array"""
    if len(arr) == 0:
        return None
    head = ListNode()
    temp = head
    for item in arr:
        temp.next = ListNode(item)
        temp = temp.next
    return head.next

def print_list(head: Optional[ListNode]) -> None:
    while head:
        print(head.val, end=' ')
        head = head.next

def reverse_list(head: Optional[ListNode]) -> Optional[ListNode]:
    """Reverse Linked list"""
    # iterative way
    # if head is None or head.next is None:
    #     return head
    # prev = None
    # curr = head.next
    # while head:
    #     head.next = prev
    #     prev = head
    #     head = curr
    #     if curr:
    #         curr = curr.next
    # return prev

    # recursive way
    if head is None or head.next is None:
        return head
    temp = reverse_list(head.next)
    head.next.next = head
    head.next = None
    return temp
        

if __name__ == '__main__':
    arr = list(map(int, input().split()))
    head = construct_LL(arr)
    # print_list(head)
    print_list(reverse_list(head))