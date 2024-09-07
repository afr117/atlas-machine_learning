#!/usr/bin/env python3

add_arrays = __import__('4-line_up').add_arrays

arr1 = [1, 2, 3, 4]
arr2 = [5, 6, 7, 8]
print(add_arrays(arr1, arr2))  # Expected Output: [6, 8, 10, 12]
print(arr1)  # Expected Output: [1, 2, 3, 4]
print(arr2)  # Expected Output: [5, 6, 7, 8]
print(add_arrays(arr1, [1, 2, 3]))  # Expected Output: None
