#!/usr/bin/env python3

cat_arrays = __import__('6-howdy_partner').cat_arrays

arr1 = [1, 2, 3, 4, 5]
arr2 = [6, 7, 8]
print(cat_arrays(arr1, arr2))  # Expected Output: [1, 2, 3, 4, 5, 6, 7, 8]
print(arr1)  # Expected Output: [1, 2, 3, 4, 5]
print(arr2)  # Expected Output: [6, 7, 8]
