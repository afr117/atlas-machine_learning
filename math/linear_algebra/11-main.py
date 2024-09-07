#!/usr/bin/env python3

import numpy as np
np_transpose = __import__('11-the_western_exchange').np_transpose

mat1 = np.array([1, 2, 3, 4, 5, 6])
mat2 = np.array([])
mat3 = np.array([[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]],
                 [[11, 12, 13, 14, 15], [16, 17, 18, 19, 20]]])
print(np_transpose(mat1))  # Expected Output: [1 2 3 4 5 6]
print(mat1)  # Original matrix should remain unchanged
print(np_transpose(mat2))  # Expected Output: []
print(mat2)  # Original matrix should remain unchanged
print(np_transpose(mat3))  # Expected Output: Transposed 3D array
print(mat3)  # Original matrix should remain unchanged
