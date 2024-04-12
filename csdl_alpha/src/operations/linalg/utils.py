def process_matA_vecb(A,b):
    """
    Process matrix A and vector b for:
    - matvec
    - solve_linear
    """
    # checks:
    # - A must be 2D
    # - x must be 2D (if 1D, reshaped to 2D)
    # - A.shape[1] == x.shape[0]
    if len(A.shape) != 2:
        raise ValueError(f"Matrix A must be 2D, but has shape {A.shape}")
    if len(b.shape) == 1:
        b = b.reshape((b.size, 1))
    elif len(b.shape) == 2:
        if b.shape[1] != 1:
            raise ValueError(f"x must have one column, but has shape {b.shape}")
    elif len(b.shape) != 2:
        raise ValueError(f"Vector x must be 1D or 2D, but has shape {b.shape}")
    if A.shape[1] != b.shape[0]:
        raise ValueError(f"Number of columns of A must be equal to the number of rows of x. {A.shape[1]} != {b.shape[0]}")
    return A, b