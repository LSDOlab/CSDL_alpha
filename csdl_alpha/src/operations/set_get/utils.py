import numpy as np
def check_and_process_out_of_bound_slice(slices, shape):
    """
    slices: list of slices or 1 slice

    Returns a tuple of slices and arrays.
    - Each slice is processed to have an integer start, stop, and step within bounds.

    Throws an error if:
    - The slice is not within bounds
    - More slices defined than dimensions of the tensor
    """
    
    slice_list = []
    
    num_dims = len(shape)
    if len(slices) > num_dims:
        raise IndexError(f'Too many dimensions indexed {len(slices)} for shape {shape}')

    # For each dimension given, we have to make sure that it is within bounds
    for (dim, sl) in enumerate(slices):
        dim_size = shape[dim]
        
        if isinstance(sl, slice): # slice

            # Replaces the Nones in the slice with the start, stop, and step values
            start, stop, step = sl.indices(dim_size)
            current_slice = slice(start, stop, step)
            slice_list.append(current_slice)

            # Check if the slice is within bounds
            if start < 0 or stop > dim_size:
                raise IndexError(f'Slice {sl} out of bounds for dimension {dim} with size {dim_size}.')
        elif isinstance(sl, list): # list of indices
            if max(sl) >= dim_size or min(sl) < 0:
                raise IndexError(f'Index {sl} out of bounds for dimension {dim} with size {dim_size}.')
            slice_list.append(sl)
        else: # integer
            if sl >= shape[dim] or sl < 0:
                raise IndexError(f'Index {sl} out of bounds for dimension {dim} with size {dim_size}.')
            slice_list.append(sl)
            
    return tuple(slice_list)