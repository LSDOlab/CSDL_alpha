import csdl_alpha.utils.testing_utils as csdl_tests
import pytest

class TestDerivative(csdl_tests.CSDLTest):
    
    def test_diag_correctness(self,):
        self.prep()

        import csdl_alpha as csdl
        import numpy as np

        recorder = csdl.build_new_recorder(inline = True)
        recorder.start()
        x_val = np.arange(6).reshape((3,2))+2.0
        y_val = np.arange(6).reshape((3,2))/2.0+3.0

        x = csdl.Variable(name = 'x', value = x_val)
        y = csdl.Variable(name = 'y', value = y_val)

        z = y*x**y
        w = z**2+2*y

        deriv = csdl.derivative([z, w], [x,y])

        dz_dx = deriv[z, x]
        dz_dy = deriv[z, y]
        dw_dx = deriv[w, x]
        dw_dy = deriv[w, y]

        deriv_diag = csdl.derivative([z, w], [x,y], elementwise=True)
        dz_dx_diag = deriv_diag[z, x]
        dz_dy_diag = deriv_diag[z, y]
        dw_dx_diag = deriv_diag[w, x]
        dw_dy_diag = deriv_diag[w, y]

        assert np.array_equal(dz_dx.value, dz_dx_diag.value)
        assert np.array_equal(dz_dy.value, dz_dy_diag.value)
        assert np.array_equal(dw_dx.value, dw_dx_diag.value)
        assert np.array_equal(dw_dy.value, dw_dy_diag.value)

    def test_diag_invalid(self,):
        self.prep()

        import csdl_alpha as csdl
        import numpy as np

        recorder = csdl.build_new_recorder(inline = True)
        recorder.start()
        x_val = np.arange(6).reshape((3,2))+2.0
        y_val = np.arange(6).reshape((3,2))/2.0+3.0

        x = csdl.Variable(name = 'x', value = x_val)
        y = csdl.Variable(name = 'y', value = y_val)

        z = y*x**y+x[0,0] + y[0,0]
        w = z**2+2*y

        deriv = csdl.derivative([z, w], [x,y])

        dz_dx = deriv[z, x]
        dz_dy = deriv[z, y]
        dw_dx = deriv[w, x]
        dw_dy = deriv[w, y]

        deriv_diag = csdl.derivative([z, w], [x,y], elementwise=True)
        dz_dx_diag = deriv_diag[z, x]
        dz_dy_diag = deriv_diag[z, y]
        dw_dx_diag = deriv_diag[w, x]
        dw_dy_diag = deriv_diag[w, y]

        assert not np.array_equal(dz_dx.value, dz_dx_diag.value)
        assert not np.array_equal(dz_dy.value, dz_dy_diag.value)
        assert not np.array_equal(dw_dx.value, dw_dx_diag.value)
        assert not np.array_equal(dw_dy.value, dw_dy_diag.value)

    def test_diag_error(self,):
        self.prep()

        import csdl_alpha as csdl
        import numpy as np

        recorder = csdl.build_new_recorder(inline = True)
        recorder.start()
        x_val = np.arange(6).reshape((3,2))+2.0
        y_val = np.arange(6).reshape((3,2))/2.0+3.0

        x = csdl.Variable(name = 'x', value = x_val)
        y = csdl.Variable(name = 'y', value = y_val)

        z = x[0,0] + y[0,0]
        w = z**2+2*y

        with pytest.raises(ValueError):
            deriv = csdl.derivative([z], [x,y], elementwise=True)
        with pytest.raises(ValueError):
            deriv = csdl.derivative([w[0,0]], [x,y], elementwise=True)
            
if __name__ == '__main__':
    test = TestDerivative()
    test.test_diag_correctness()
    test.test_diag_invalid()
    test.test_diag_error()