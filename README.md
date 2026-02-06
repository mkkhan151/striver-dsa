# Striver's DSA Course - Python Solutions

Python implementations of problems from [Striver's DSA Sheet](https://takeuforward.org/strivers-a2z-dsa-course/strivers-a2z-dsa-course-sheet-2), a popular roadmap for mastering Data Structures and Algorithms.

## Repository Structure

```
.
├── arrays_problems.py
├── binary_search_problems.py
├── string_problems.py
├── sorting.py
├── basic_recursion.py
├── backtracking_problems.py
├── armstrong_number.py
├── main.py
├── linked_list/
│   ├── linked_list.py
│   ├── doubly_linked_list.py
│   └── linked_list_problems.py
└── graph/
    ├── learning.py
    ├── traversal.py
    └── dfs_bfs_problems.py
```

## Topics Covered

| Topic | Problems | File |
|-------|----------|------|
| Arrays (Easy, Medium, Hard) | 41 | `arrays_problems.py` |
| Binary Search (1D, 2D, Answer) | 24 | `binary_search_problems.py` |
| Graphs (DFS, BFS, Components) | 27 | `graph/dfs_bfs_problems.py` |
| Linked Lists (Singly, Doubly) | 19 | `linked_list/linked_list_problems.py` |
| Strings | 15 | `string_problems.py` |
| Recursion | 8 | `basic_recursion.py` |
| Sorting Algorithms | 7 | `sorting.py` |
| Backtracking | 5 | `backtracking_problems.py` |
| **Total** | **147+** | |

## Highlights

### Arrays
Covers easy to hard problems including two sum, Kadane's algorithm, next permutation, Pascal's triangle, 3-sum/4-sum, merge intervals, count inversions, and more.

### Binary Search
1D search problems, rotated sorted arrays, peak elements, square root, Nth root, book allocation, split array, and 2D matrix search problems.

### Graphs
BFS/DFS traversals, cycle detection, number of islands, rotting oranges, flood fill, surrounded regions, Pacific Atlantic water flow, and binary tree level-order problems.

### Linked Lists
Reversal, cycle detection, palindrome check, merge sort, intersection detection, reverse K-group, deep copy with random pointers, and reorder list.

### Backtracking
Word search, letter combinations, subset generation, valid parentheses generation, and combination sum.

### Sorting
Selection sort, bubble sort, insertion sort, merge sort, quick sort — including recursive variants.

## Notes

- Most solutions include multiple approaches with time and space complexity analysis in comments.
- Functions use `snake_case`, classes use `PascalCase`.
- I/O is handled via `sys.stdin`/`sys.stdout` redirection to `input.txt` and `output.txt` in some files.
