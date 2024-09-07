#!/usr/bin/env python3

add_matrices2D = __import__('5-across_the_planes').add_matrices2D

mat1 = [[1, 2], [3, 4]]
mat2 = [[5, 6], [7, 8]]
print(add_matrices2D(mat1, mat2))  # Expected Output: [[6, 8], [10, 12]]
print(mat1)  # Expected Output: [[1, 2], [3, 4]]
print(mat2)  # Expected Output: [[5, 6], [7, 8]]
print(add_matrices2D(mat1, [[1, 2, 3], [4, 5, 6]]))  # Expected Output: None
