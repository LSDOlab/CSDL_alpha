import numpy as np

class Matrix():
    '''
    Base class used to store a matrix.

    Attributes
    ----------
    shape : tuple
        Shape of the matrix
    data : np.ndarray
        Numpy array that stores the matrix
    '''

    def __init__(self, shape, x=None):
        '''
        Initialize the matrix class

        Parameters
        ----------
        shape : tuple
            Shape of the matrix
        x : np.ndarray
            Initial value for the matrix
        '''
        self.shape = shape
        if x is None:
            self.data = np.zeros(shape)
        else:
            self.data=x