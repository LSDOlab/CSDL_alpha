class Slice():
    def __getitem__(self, key):
        return key
    
_slice = Slice()

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