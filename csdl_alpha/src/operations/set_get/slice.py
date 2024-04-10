class Slicer():
    def __init__(self):
        """CSDL's slice object. Used for setting values of array tensors
        
        build slices by using the slice object and indexing with it: slice[...]

        slice indices must be: int, slice, list[int], tuple[int, slice, list[int]]
            examples of valid slices:
            - slice[1]
            - slice[1:, 0:3]
            - slice[3:, 2, :]
            - slice[1, 2, 3]
            - slice[[1, 2, 3], [0, 1, 2]]

            examples of invalid slices:
            - slice[..., 1]

        Slicing Rules for CSDL:
        1) CSDL variables should not be used as indicies if not in a for loop.
        
        """
        import numpy as np
        self.valid_types = (slice, int, np.integer, list)

    def __getitem__(self, keys):
        """
        Parameters
        ----------
        key : int, slice, tuple[int, slice, tuple[tuple[int]]], tuple[tuple[int]]
            examples of valid slices:
            - slice[1]
            - slice[1:, 0:3]
            - slice[3:, 2, :]
            - slice[1, 2, 3]
            - slice[5:10, [1, 2, 3], [0, 1, 2], 5]
            - slice[[1, 2, 3]]

            examples of invalid slices:
            - slice[..., 1]
            - slice[0:1,[0,1],[0,1],1, [1,2,3], [1,2,3], 0:1]

        Returns
        -------
        key
        """

        # tuplify keys if not already a tuple
        if isinstance(keys, self.valid_types):
            keys = (keys,)
        
        # check to make sure all axes are correct types and dimensions
        return_keys = []
        current_list_length = None
        for k in keys:
            # check to make sure types are correct
            current_list_length = self.check_axis_validity(k, return_keys, current_list_length)

        return tuple(return_keys)

    def check_axis_validity(
            self,
            k,
            return_keys,
            current_list_length,
        ):
        """
        Check if the key is a valid type and if the list of index sets is valid.
        """
        if not isinstance(k, self.valid_types):
            raise TypeError(f"Invalid key {k}. Must be an integer, slice, or tuple index list.")
        
        # If a list of index sets but only one element, then just set it as an integer
        if isinstance(k, list) and len(k) == 1:
            return_keys.append(k[0])
        else:
            return_keys.append(k)
        
        if not isinstance(k, list):
            if current_list_length is not None:
                current_list_length = -1
        elif len(k) > 1: 
            # if current_list_length is None, then we have not seen a list of index sets yet
            # If current_list_length is -1, then we have already seen a list of index sets

            # if k is a list, we need to make sure there is only one contiguous sequence of lists. 
            # current_list_length is the length of the first list which must be the same for all lists after.

            if current_list_length == -1:
                raise IndexError(f"Invalid key {k}. Only 1 contiguous sequence of index set lists allowed.")
            if current_list_length is None:
                current_list_length = len(k)
            elif current_list_length != len(k):
                raise IndexError(f"Invalid key {k}. Expected list of length {current_list_length}, length {len(k)} given.")

        return current_list_length


_slice = Slicer()

class Slice():
    def __init__(self, keys:tuple):
        self.tuple = keys

        for i, k in enumerate(keys):
            if isinstance(k, list):
                keys[i] = tuple(k)


def get_slice_shape(s, parent_shape):
    '''
    Get the shape of the slice s of a tensor with shape parent_shape.

    Arguments
    ---------
    s : tuple
        Slice can be a tuple of slices or a single slice or list of index sets.
    parent_shape : tuple
        Shape of the parent tensor.
    '''
    import numpy as np
    slice_shape = np.asarray(parent_shape)

    # if slice along the first axis of a tensor or a single index on the first axis
    # (shape must be () for a single index in tuple form)
    if not isinstance(s, tuple):
        s = (s,)

    # if list of index sets
    if all(isinstance(sl, tuple) for sl in s):
        slice_shape = np.array([len(s[0]),])
    # if has at least one slice
    else:
        delete_dims = []
        for (dim, sl) in enumerate(s):
            if isinstance(sl, slice):
                start, stop, step = sl.indices(parent_shape[dim])
                slice_shape[dim] = (stop - start) // step
            else: # isinstance(sl, tuple):
                if isinstance(sl, int):
                    delete_dims.append(dim)
                else:
                    slice_shape[dim] = len(s[dim])

        if len(delete_dims) > 0:
            slice_shape = np.delete(slice_shape, delete_dims)    
    
    return tuple(slice_shape)

if __name__ == '__main__':
    import numpy as np

    x = np.ones((10,10,10))

    slices = (slice(0, 5), (1,2) , (2,3))
    print(get_slice_shape(slices, x.shape))
    print(x[slices].shape)