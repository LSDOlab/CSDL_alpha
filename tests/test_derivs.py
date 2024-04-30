import csdl_alpha as csdl
import numpy as np
import csdl_alpha.utils.testing_utils as csdl_tests

class TestDeriv(csdl_tests.CSDLTest):
    def test_deriv(self):
        self.prep()

        # test basics
        x0 = csdl.Variable(value=np.arange(6).reshape(3,2)+1.0)
        x1 = csdl.Variable(value=2.0)

        y = csdl.norm(x0[[0,1], :]) + csdl.sin(x1)
        print(np.sin(2.0))
        print(np.cos(2.0))
        print(np.linalg.norm(np.sin(2.0)))

        dy_dx = csdl.derivative.reverse(y, [x0, x1])
        dy_dx0 = dy_dx[x0]
        dy_dx1 = dy_dx[x1]

        compare_values = []
        y_real = np.linalg.norm(np.arange(4).reshape(2,2)+1.0) + np.sin(2.0)
        compare_values += [csdl_tests.TestingPair(y, y_real.reshape(1,))]
        compare_values += [csdl_tests.TestingPair(dy_dx1, np.cos(2.0).reshape(1,1))]

        self.run_tests(compare_values=compare_values, verify_derivatives=True)

if __name__ == '__main__':
    t = TestDeriv()
    t.test_deriv()





