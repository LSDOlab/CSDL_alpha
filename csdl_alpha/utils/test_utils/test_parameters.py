from csdl_alpha.utils.testing_utils.csdl_test import CSDLTest
import csdl_alpha as csdl
import pytest
import numpy as np

def check_valid(name, value):
    if not int(np.sqrt(value)) == np.sqrt(value):
        raise ValueError(f"{name} must be a perfect square.")


class TestParameters(CSDLTest):
    def test_wrong_inputs(self):
        with pytest.raises(TypeError):
            csdl.check_parameter(1, 'a', values=12)

        with pytest.raises(TypeError):
            csdl.check_parameter(1, 'a', types='int, float')

        with pytest.raises(RuntimeError):
            csdl.check_parameter(1, 'a', values=(1,2,3), types=(int, float))

    def test_correct_inputs(self):
        csdl.check_parameter(1, 'a', values=(1,2,3))
        csdl.check_parameter(1, 'a', 
                             types=(int, float), 
                             check_valid=check_valid,
                             upper=3, lower=1)
        csdl.check_parameter(None, 'a', 
                             types=(int, float),
                             upper=3, lower=1, 
                             allow_none=True)
        csdl.check_parameter(True, 'a', types=bool)
        
    def test_invalid_value(self):
        with pytest.raises(TypeError):
            csdl.check_parameter('a', 'a', 
                                types=(int, float))
        with pytest.raises(TypeError):
            csdl.check_parameter(4, 'a', 
                                types=str)
        with pytest.raises(ValueError):
            csdl.check_parameter(3, 'a', 
                                check_valid=check_valid)
        with pytest.raises(ValueError):
            csdl.check_parameter(9, 'a', 
                                upper=3, lower=1)
        with pytest.raises(ValueError):
            csdl.check_parameter(0, 'a', 
                                upper=3, lower=1)
        with pytest.raises(ValueError):
            csdl.check_parameter(4, 'a', 
                                values=(1,2,3))
        with pytest.raises(ValueError):
            csdl.check_parameter('a', 'a', 
                                values=(1,2,3))