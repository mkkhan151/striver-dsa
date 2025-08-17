from typing import Any, Dict, Optional
import sys

sys.stdin = open('input.txt')
sys.stdout = open('output.txt', 'w')


class Node:
    def __init__(self, x: int, next: 'Node' = None, random: 'Node' = None):
        self.val = int(x)
        self.next = next
        self.random = random

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
        
def middle_node(head: Optional[ListNode]) -> Optional[ListNode]:
    """Returns middle node of linked list"""
    # Approach 1: TC -> O(2n)
    # count = 0
    # temp = head
    # while temp:
    #     count += 1
    #     temp = temp.next
    # m = count // 2
    # mid = head
    # while m > 0 and mid:
    #     mid = mid.next
    #     m -= 1
    # return mid

    # Approach 2: TC -> O(n/2)
    curr = mid = head
    while curr and curr.next and mid:
        curr = curr.next.next
        mid = mid.next
    return mid

def has_cycle(head: Optional[ListNode]) -> bool:
    """Returns true if linked list has cycle otherwise false"""
    slow = fast = head
    while fast and fast.next and slow:
        fast = fast.next.next
        slow = slow.next
        if fast is slow:
            return True
    return False

def detect_cycle(head: Optional[ListNode]) -> Optional[ListNode]:
    """Returns the node where cycle begins other wise null"""
    # method 1: using hashmap
    # visited = set()
    # temp = head
    # while temp:
    #     if temp in visited:
    #         return temp
    #     else:
    #         visited.add(temp)
    #     temp = temp.next
    # return None

    # method 2: using hare and tortoise algorithm
    if head is None or head.next is None:
        return head
    
    slow = fast = head
    while fast and fast.next and slow:
        fast = fast.next.next
        slow = slow.next

        if slow == fast:
            break
    else:
        return None
    
    slow = head
    while slow != fast and slow:
        slow = slow.next
        fast = fast.next
    return slow

def count_nodes_in_loop(head):
    """Return the count of nodes in cycle/loop of linked list if exist"""
    if head is None or head.next is None:
        return 0
    
    slow = fast = head
    while fast and fast.next and slow:
        fast = fast.next.next
        slow = slow.next

        if slow == fast:
            break
    else:
        return 0
    
    count = 1
    slow = slow.next
    while slow != fast:
        slow = slow.next
        count += 1
    return count

def is_palindrome(head: Optional[ListNode]) -> bool:
    """Returns True if linked list is palindrome otherwise False"""
    # method 1
    # if head is None or head.next is None:
    #     return False
    # stack = []
    # slow = fast = head
    # while fast and fast.next and slow:
    #     stack.append(slow.val)
    #     slow = slow.next
    #     fast = fast.next.next
    
    # if fast and slow:
    #     slow = slow.next
    # while stack and slow:
    #     if stack.pop() == slow.val:
    #         slow = slow.next
    #     else:
    #         return False
    # return True

    # method 2
    if head is None:
        return False
    slow = fast = head
    while fast and fast.next and slow:
        slow = slow.next
        fast = fast.next.next
    
    if fast and slow:
        slow = slow.next
    # reverse the half of the list
    fast = reverse_list(slow)
    slow = head
    while fast and slow:
        if fast.val == slow.val:
            fast = fast.next
            slow = slow.next
        else:
            return False
    return True

def odd_even_list(head: Optional[ListNode]) -> Optional[ListNode]:
    """Separate odd and even indices nodes and return in odd, even order"""
    if head is None or head.next is None or head.next.next is None:
        return head
    
    odd: ListNode = head
    even: ListNode = head.next

    odd_p = odd
    even_p = even
    temp: ListNode | None = head.next.next
    while temp:
        odd_p.next = temp
        odd_p = odd_p.next
        temp = temp.next

        if temp:
            even_p.next = temp
            even_p = even_p.next
            temp = temp.next
    odd_p.next = even
    even_p.next = None
    
    return odd

def remove_nth_from_end(head: Optional[ListNode], n: int) -> Optional[ListNode]:
    """Remove the Nth node from end of the linked list and return its head"""

    # calculate size
    # temp = head
    # size = 0
    # while temp:
    #     temp = temp.next
    #     size += 1
    # # calculate index from head
    # n = size - n

    # if n == 0:
    #     temp = head
    #     head = head.next
    #     del temp
    #     return head

    # temp = head
    # while n > 1 and temp:
    #     temp = temp.next
    #     n -= 1

    # d = temp.next
    # temp.next = d.next
    # del d
    # return head

    slow = fast = head
    while n > 0 and fast:
        fast = fast.next
        n -= 1

    if fast is None:
        return head.next
    
    while fast.next:
        slow = slow.next
        fast = fast.next
    d = slow.next
    slow.next = d.next
    del d
    return head

def merge_two_lists(list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
    """Merge two sorted lists into one and returns its head"""
    if list1 is None and list2 is None:
        return None
    
    head = ListNode(-101)
    curr = head
    while list1 and list2:
        if list1.val < list2.val:
            curr.next = list1
            list1 = list1.next
        else:
            curr.next = list2
            list2 = list2.next
        curr = curr.next
    if list1:
        curr.next = list1
    else:
        curr.next = list2
    return head.next
        
def reorder_list(head: Optional[ListNode]) -> None:
    """Reorders the list"""
    if head is None or head.next is None:
        return
    slow, fast = head, head.next
    # find middle point of list
    while fast and fast.next and slow:
        slow = slow.next
        fast = fast.next.next

    second = slow.next
    slow.next = None
    # reverse right half list
    second = reverse_list(second)
    first = head
    while second:
        tmp1, tmp2 = first.next, second.next
        first.next = second
        second.next = tmp1
        first, second = tmp1, tmp2

def copy_random_list(head: Optional[Node]) -> Optional[Node]:
    """Returns the deep copy of random pointer list"""
    old_to_copy: Dict[Node | None, Node | None] = {None : None}
    curr = head
    while curr:
        copy = Node(curr.val)
        old_to_copy[curr] = copy
        curr = curr.next

    curr = head
    while curr:
        copy = old_to_copy[curr]
        copy.next = old_to_copy[curr.next]
        copy.random = old_to_copy[curr.random]
        curr = curr.next

    return old_to_copy[head]
    
def add_two_numbers(l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
    """
    Adds two decimal numbers given in the two linked lists 
    each node representing a single digit in a number
    returns the sum in linked list form
    """
    if not l1 and not l2:
        return None
    carry: int = 0
    result_head = ListNode(-1)
    curr = result_head
    while l1 and l2:
        sum = l1.val + l2.val + carry
        carry = sum // 10
        val = sum % 10
        curr.next = ListNode(val)
        curr = curr.next
        l1 = l1.next
        l2 = l2.next
    while l1:
        sum = l1.val + carry
        carry = sum // 10
        val = sum % 10
        curr.next = ListNode(val)
        curr = curr.next
        l1 = l1.next
    while l2:
        sum = l2.val + carry
        carry = sum // 10
        val = sum % 10
        curr.next = ListNode(val)
        curr = curr.next
        l2 = l2.next
    if carry != 0:
        curr.next = ListNode(carry)
    return result_head.next

def delete_middle(head: Optional[ListNode]) -> Optional[ListNode]:
    """ Delete the middle node of the list and return head """

    # Brute force
    # if not head:
    #     return head
    # if not head.next:
    #     del head
    #     return None
    
    # size = 0
    # curr = head
    # while curr:
    #     size += 1
    #     curr = curr.next
    
    # mid = size // 2
    # curr = head
    # while mid > 1 and curr:
    #     mid -= 1
    #     curr = curr.next
    # temp = curr.next
    # curr.next = temp.next
    # temp.next = None
    # del temp
    # return head

    # optimize
    if not head:
        return head
    # delete the only node in the list b.c n=1 => mid=0
    if not head.next:
        del head
        return None
    
    slow = fast = head
    prev_slow = None
    while fast and fast.next and slow:
        prev_slow = slow
        slow = slow.next
        fast = fast.next.next
    prev_slow.next = slow.next
    slow.next = None
    del slow
    return head
    
def sort_list(head: Optional[ListNode]) -> Optional[ListNode]:
    """ Return the sorted list in ascending order """

    if not head or not head.next:
        return head
    # using insertion method
    # sorted_list = head
    # head = head.next
    # sorted_list.next = None

    # while head:
    #     curr = head
    #     head = head.next
        
    #     if curr.val < sorted_list.val:
    #         curr.next = sorted_list
    #         sorted_list = curr
    #     else:
    #         temp = sorted_list
    #         while temp and temp.next and curr.val > temp.next.val:
    #             temp = temp.next
    #         curr.next = temp.next
    #         temp.next = curr
    # return sorted_list

    # using merge sort
    # Divide list into two halves
    slow = fast = head
    # skip one step of slow ptr for better division of list
    fast = fast.next.next
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    fast = slow.next
    slow.next = None

    return merge_two_lists(sort_list(head), sort_list(fast))

def get_intersection_node(headA: Optional[ListNode], headB: Optional[ListNode]) -> Optional[ListNode]:
    """ Returns the node where two lists intersect each other else returns None """
    if not headA or not headB:
        return None
    
    # Brute force O(n * m)
    # while headA:
    #     temp = headB
    #     while temp:
    #         if temp == headA:
    #             return head.next
    #         temp = temp.next
    #     headA = headA.next
    # return None

    #  using hash map O(n + m), O(n)
    # visited = set()
    # while headA:
    #     visited.add(headA)
    #     headA = headA.next
    # while headB:
    #     if headB in visited:
    #         return headB
    #     headB = headB.next
    # return None

    # More Optimized
    sizeA = sizeB = 0
    tempA = headA
    tempB = headB

    while tempA:
        sizeA += 1
        tempA = tempA.next
    while tempB:
        sizeB += 1
        tempB = tempB.next
    
    if sizeA > sizeB:
        skip = sizeA - sizeB
        while skip > 0:
            headA = headA.next
            skip -= 1
    else:
        skip = sizeB - sizeA
        while skip > 0:
            headB = headB.next
            skip -= 1

    while headA and headB:
        if headA == headB:
            return headA
        headA = headA.next
        headB = headB.next
    return None

if __name__ == '__main__':
    # arr = list(map(int, input().split()))
    # head = construct_LL(arr)
    # print_list(head)
    # print_list(reverse_list(head))
    # print_list(middle_node(head))
    # print(is_palindrome(head))
    # print_list(odd_even_list(head))
    # print_list(remove_nth_from_end(head, 5))
    # reorder_list(head)
    # print_list(head)

    # num1 = list(map(int, input().split()))
    # num2 = list(map(int, input().split()))
    # num1 = construct_LL(num1)
    # num2 = construct_LL(num2)
    # print_list(add_two_numbers(num1, num2))

    arr = list(map(int, input().split()))
    head = construct_LL(arr)
    # print_list(delete_middle(head))
    print_list(sort_list(head))