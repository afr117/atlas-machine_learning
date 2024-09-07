#!/usr/bin/env python3

cat_matrices2D = __import__('7-gettin_cozy').cat_matrices2D

mat1 = [[1, 2], [3, 4]]
mat2 = [[5, 6]]
mat3 = [[7], [8]]
mat4 = cat_matrices2D(mat1, mat2)  # Concatenating along axis=0 (rows)
mat5 = cat_matrices2D(mat1, mat3, axis=1)  # Concatenating along axis=1 (columns)
print(mat4)  # Expected Output: [[1, 2], [3, 4], [5, 6]]
print(mat5)  # Expected Output: [[1, 2, 7], [3, 4, 8]]
mat1[0] = [9, 10]
mat1[1].append(5)
print(mat1)  # Expected Output: [[9, 10], [3, 4, 5]]
print(mat4)  # Expected Output: [[1, 2], [3, 4], [5, 6]]
print(mat5)  # Expected Output: [[1, 2, 7], [3, 4, 8]]
