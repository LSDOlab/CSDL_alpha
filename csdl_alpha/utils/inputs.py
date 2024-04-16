import numpy as np
from .error_utils.error_utils import check_if_valid_shape

def get_type_string(obj)-> str:
    return f'\'{type(obj).__name__}\''

def ingest_value(value, dtype=np.float64):
    if isinstance(value, (float, int, np.integer, np.floating)):
        value = np.array([value], dtype=dtype)
    elif isinstance(value, np.ndarray):
        value = value.astype(dtype)
    elif value is not None:
        raise TypeError(f"Value must be a numpy array, float or int. Value {value} of type {get_type_string(value)} given")
    return value

def scalarize(value):
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return value[0]
        else:
            raise ValueError(f"Value must be a scalar. {value.shape} given")
    elif not isinstance(value, (float, int)):
        raise ValueError(f"Value must be a scalar. {value} given")
    else:
        return value


def process_shape_and_value(shape, value, dtype=np.float64):
    value = ingest_value(value, dtype=dtype)
    if shape is not None:
        check_if_valid_shape(shape)
        if value is not None and shape != value.shape:
            if value.shape == (1,):
                value = value[0]*np.ones(shape)
            else:
                raise ValueError("Shape and value shape must match")
    elif value is None:
        raise ValueError("Shape or value must be provided")
    else:
        shape = value.shape
    return shape, value


def validate_and_variablize(value, raise_on_sparse = True):
    """Must be called on all variables that are inputs to operations

    Parameters
    ----------
    value : _type_
        _description_
    raise_on_sparse : bool, optional
        _description_, by default True

    Returns
    -------
    _type_
        _description_

    Raises
    ------
    ValueError
        _description_
    """

    var = variablize(value)

    from csdl_alpha.src.graph.variable import SparseMatrix
    if isinstance(var, SparseMatrix):
        if raise_on_sparse:
            raise TypeError("Sparse matrices not supported for this value.")
    return var

def variablize(variable):
    from csdl_alpha.src.graph.variable import Variable
    if isinstance(variable, Variable):
        return variable
    else:
        var = Variable(value = ingest_value(variable))
        return var

def get_shape(shape, value):
    if shape is None:
        if value is not None:
            shape = value.shape
        else:
            raise ValueError("Shape or value must be provided")
    else:
        check_if_valid_shape(shape)
        if value is not None:
            if shape != value.shape:
                raise ValueError("Shape and value shape must match")
    return shape