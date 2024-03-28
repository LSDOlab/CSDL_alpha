
# TODO: add indexing operation.
# Notes:
"""
Numpy indexing behaviour:
- Integer indices contract:
-- if A.shape = (2,3,4,5), then A[0] is a (3x4x5) array
- Slicing does not contract:
-- if A.shape = (2,3,4,5), then A[0:1] is a (1x3x4x5) array
- negative indices are only allowed from -n to +(n-1) where n is the size of 
  the dimension minus 1
-- the the mapping:
       -n --> 0,
       -(n-1) --> 1,
       ...,
       -1 --> n-1
-- A[-4] throws an index out of bounds error

slices:
- automatically created in set_item and get_item if colon.
- Slice can be represented by a tuple of (start, stop, step)
- if step is negative, then start should be greater than stop

raise error if a slice is empty 
- ie i:i


========Implementation========
recursion:
def index(x, remaining_slices):
   if len(remaining_slices) == 0:
       return x

   else:
       current_slice = remaining_slices[0]
       
       x = np.empty()
       for indices in current_slice:
           x.stack(index(x, indices))
       
       index(x[current_slice], [remaining_slices[1:])

   if array:
   if int:




"""

def slice_to_tuple(key: slice, size: int) -> tuple:
    if key.start is None:
        key_start = 0
    else:
        key_start = key.start

    if key.stop is None:
        key_stop = size
    else:
        key_stop = key.stop
    
    return key_start, key_stop, key.step

if __name__ == '__main__':
    import numpy as np
    # def index(A, remaining_slices, current_dim):
    #     current_slice = remaining_slices[0]
    #     dim = -len(remaining_slices)
    #     # start, stop, step = slice_to_tuple(current_slice, A.shape[0])

    #     if len(remaining_slices) == 1:
    #         # for 
    #         return A[current_slice]
    #     else:
    #         return index(A, remaining_slices[1:])

    #     return A


    # def index(A, remaining_slices,current_dim):
    #     while len(remaining_slices) > 0:
    #         current_slice = remaining_slices[0]
    #         prefix = [slice(0,None,None) for i in range(current_dim)] + [current_slice]
    #         A = A[tuple(prefix)]
    #         print(A)
    #         remaining_slices = remaining_slices[1:]

    #         current_dim+=1
    #     return A
    # A = np.arange(120).reshape(2,3,4,5)
    # # A = np.arange(12).reshape(3,4)
    # print(A)
    # print()
    # slice_tuple = (slice(0,2,1),)
    # slice_tuple = (slice(0,2,1),1,)
    # slice_tuple = (slice(0,2,1),slice(0,2,1),slice(0,2,1))

    # A_sliced = index(A, slice_tuple, 0)
    # print()
    # A_numpy_slice = A[slice_tuple]
    # print(A_sliced)
    # print(A_numpy_slice)
    # print(np.linalg.norm(A_numpy_slice - A_sliced))



    shapes = (10, 10,10, 10,10, 10,10, 10,10, 10,10, 10,10)
    A = np.arange(np.prod(shapes)).reshape(*shapes)

    import time

    start = time.time()
    A[:,:,:,1:9,:,:,1:9,1:9,1:9]
    end = time.time()
    print(end-start)
