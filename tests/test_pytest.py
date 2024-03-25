import pytest
import numpy as np


def add_two_integers(x,y):
    
    if not isinstance(x, int):
        raise TypeError(f'{x} is not an integer.')
    if not isinstance(y, int):
        raise TypeError(f'{y} is not an integer.')
    return x+y

'''
Test to make sure desired output is correct
'''
def test_sample_add_1():
    '''
    Test description: make sure integers are added correctly.
    '''

    # Import class/function to test

    # Run test scenario
    input_val = 3
    desired_val = 6
    actual_val = add_two_integers(
        input_val, 
        input_val
    )

    # Check to make sure values are correct
    np.testing.assert_almost_equal(
        actual_val, 
        desired_val,
        decimal = 7,
    )

'''
Test to make sure desired output is correct
'''
def test_sample_add_2():
    '''
    Test description: make sure integers are added correctly.
    '''

    # Import class/function to test

    # Run test scenario
    input_val = 4
    desired_val = 8
    actual_val = add_two_integers(
        input_val, 
        input_val
    )

    # Check to make sure values are correct
    assert actual_val == desired_val


'''
Test to make sure exceptions are raised
'''
def test_add_exception():

    '''
    Test description: if input is not an integer, error should be raised
    '''

    # Import class/function to test

    # Check to make sure exceptions are raised
    with pytest.raises(Exception) as exc_info:

        # Run test scenario
        input_val = 4.5
        actual_val = add_two_integers(
            input_val, 
            input_val
        )

    # Make sure the correct errors are being raised
    assert exc_info.type is TypeError
    assert str(exc_info.value) == '4.5 is not an integer.'



