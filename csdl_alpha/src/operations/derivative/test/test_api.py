import csdl_alpha.utils.testing_utils as csdl_tests

class TestDerivative(csdl_tests.CSDLTest):
    
    def test_output_datatype(self,):
        self.prep()

        import csdl_alpha as csdl
        import numpy as np

        recorder = csdl.build_new_recorder(inline = True)
        recorder.start()
        x_val = np.array([1.0])
        y_val = 3.0

        x = csdl.Variable(name = 'x', value = x_val)
        y = csdl.Variable(name = 'y', value = y_val)

        z = y*x
        deriv1 = csdl.derivative([z], [x])[z,x]
        deriv3 = csdl.derivative(z, [x])[x]
        deriv2 = csdl.derivative([z], x)[z]
        deriv4 = csdl.derivative(z, x)

        assert np.array_equal(deriv1.value, deriv2.value) and np.array_equal(deriv2.value, deriv3.value) and np.array_equal(deriv3.value, deriv4.value)

        z = y*x
        w = y/x
        deriv = csdl.derivative([z,w], [x,y])
        assert isinstance(deriv[z,x], csdl.Variable)
        assert isinstance(deriv[w,y], csdl.Variable)

        deriv = csdl.derivative([z,w], [x,y], as_block = True)
        assert isinstance(deriv, csdl.Variable)
        assert deriv.shape == (2,2)

        deriv = csdl.derivative([z], [x,y], as_block = True)
        assert isinstance(deriv, csdl.Variable)
        assert deriv.shape == (1,2)

        deriv = csdl.derivative([z,w], [y], as_block = True)
        assert isinstance(deriv, csdl.Variable)
        assert deriv.shape == (2,1)

        deriv = csdl.derivative([z], [y], as_block = True)
        assert isinstance(deriv, csdl.Variable)
        assert deriv.shape == (1,1)

        deriv = csdl.derivative(z, [y, x], as_block = True)
        assert isinstance(deriv, csdl.Variable)
        assert deriv.shape == (1,2)

        deriv = csdl.derivative([z,w], y, as_block = True)
        assert isinstance(deriv, csdl.Variable)
        assert deriv.shape == (2,1)
