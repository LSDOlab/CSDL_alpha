import numpy as np

def ingest_value(value):
    if isinstance(value, (float, int)):
        value = np.array([value])
    elif not isinstance(value, np.ndarray) and value is not None:
        raise ValueError("Value must be a numpy array, float or int")
    return value
    
def check_shape(shape, value):
    if shape is None:
        if value is not None:
            shape = value.shape
        else:
            raise ValueError("Shape or value must be provided")
    else:
        if value is not None:
            if shape != value.shape:
                raise ValueError("Shape and value shape must match")
    return shape