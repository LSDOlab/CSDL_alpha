import csdl_alpha.utils.testing_utils as csdl_tests
import pytest

class TestDerivative(csdl_tests.CSDLTest):
    
    def test_concat_correctness(self,):
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
        w = (z**2+2*y)

        deriv = csdl.derivative([z, w], [x,y])

        dz_dx = deriv[z, x]
        dz_dy = deriv[z, y]
        dw_dx = deriv[w, x]
        dw_dy = deriv[w, y]

        deriv_diag = csdl.derivative([z, w], [x,y], concatenate_ofs=True)
        dz_dx_c = deriv_diag[z, x]
        dz_dy_c = deriv_diag[z, y]
        dw_dx_c = deriv_diag[w, x]
        dw_dy_c = deriv_diag[w, y]

        deriv_diag = csdl.derivative([z, w], [x,y], concatenate_ofs=True, elementwise=True)
        dz_dx_diag = deriv_diag[z, x]
        dz_dy_diag = deriv_diag[z, y]
        dw_dx_diag = deriv_diag[w, x]
        dw_dy_diag = deriv_diag[w, y]

        assert np.array_equal(dz_dx.value, dz_dx_diag.value)
        assert np.array_equal(dz_dy.value, dz_dy_diag.value)
        assert np.array_equal(dw_dx.value, dw_dx_diag.value)
        assert np.array_equal(dw_dy.value, dw_dy_diag.value)

        assert np.array_equal(dz_dx_c.value, dz_dx.value)
        assert np.array_equal(dz_dy_c.value, dz_dy.value)
        assert np.array_equal(dw_dx_c.value, dw_dx.value)
        assert np.array_equal(dw_dy_c.value, dw_dy.value)

        w0 = w[0]
        z11 = z[1,1]
        deriv_diag = csdl.derivative([z11, w0], [x,y])
        dz_dx_c = deriv_diag[z11, x]
        dz_dy_c = deriv_diag[z11, y]
        dw_dx_c = deriv_diag[w0, x]
        dw_dy_c = deriv_diag[w0, y]

        deriv_diag = csdl.derivative([z11, w0], [x,y], concatenate_ofs=True)
        dz_dx_diag = deriv_diag[z11, x]
        dz_dy_diag = deriv_diag[z11, y]
        dw_dx_diag = deriv_diag[w0, x]
        dw_dy_diag = deriv_diag[w0, y]

        assert np.array_equal(dz_dx_c.value, dz_dx_diag.value)
        assert np.array_equal(dz_dy_c.value, dz_dy_diag.value)
        assert np.array_equal(dw_dx_c.value, dw_dx_diag.value)
        assert np.array_equal(dw_dy_c.value, dw_dy_diag.value)

if __name__ == '__main__':
    test = TestDerivative()
    test.test_concat_correctness()
