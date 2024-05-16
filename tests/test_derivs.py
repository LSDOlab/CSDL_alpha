import csdl_alpha as csdl
import numpy as np
import csdl_alpha.utils.testing_utils as csdl_tests

class TestDeriv(csdl_tests.CSDLTest):
    def test_deriv(self):
        """
        Test derivatives with sequences of operations
        """
        self.prep()

        # test basics
        x0 = csdl.Variable(value=np.arange(6).reshape(3,2)+1.0)
        x1 = csdl.Variable(value=2.0)

        y = csdl.norm(x0[[0,1], :]) + csdl.sin(x1)
        # print(np.sin(2.0))
        # print(np.cos(2.0))
        # print(np.linalg.norm(np.sin(2.0)))

        dy_dx = csdl.derivative(y, [x0, x1])
        dy_dx0 = dy_dx[x0]
        dy_dx1 = dy_dx[x1]

        compare_values = []
        y_real = np.linalg.norm(np.arange(4).reshape(2,2)+1.0) + np.sin(2.0)
        compare_values += [csdl_tests.TestingPair(y, y_real.reshape(1,))]
        compare_values += [csdl_tests.TestingPair(dy_dx1, np.cos(2.0).reshape(1,1))]

        self.run_tests(compare_values=compare_values, verify_derivatives=True)

    def test_deriv_2(self):
        """
        Test second derivatives with composed operations etc
        """
        self.prep()

        # test basics
        x0 = csdl.Variable(value=np.arange(6).reshape(3,2)+1.0)
        x1 = csdl.Variable(value=2.0)
        x0.add_name('x0')
        x1.add_name('x1')

        y = csdl.norm(x0[[0,1], :]) - csdl.sin(x1)
        # y = x0+x1
        y.add_name('y')
        with csdl.Namespace('deriv1'):
            dy_dx1 = csdl.derivative(y, [x1, x0])[x1]
            dy_dx1.add_name('dy_dx1')

        with csdl.Namespace('deriv2'):
            d2y_dx12 = csdl.derivative(dy_dx1, [x1])[x1]
            d2y_dx12.add_name('d2y_dx12')

        # recorder = csdl.get_current_recorder()
        # recorder.visualize_graph()
        # exit()
        compare_values = []
        y_real = np.linalg.norm(np.arange(4).reshape(2,2)+1.0) - np.sin(2.0)
        compare_values += [csdl_tests.TestingPair(y, y_real.reshape(1,))]
        compare_values += [csdl_tests.TestingPair(d2y_dx12, d2y_dx12.value)]

        self.run_tests(compare_values=compare_values, verify_derivatives=True)

    def test_deriv_3(self):
        """
        Test more sequences with different operations
        """
        self.prep()

        x0 = csdl.Variable(value=np.arange(6).reshape(3,2)+1.0)
        x1 = csdl.Variable(value=2.0)

        y = x0.set(csdl.slice[1:2], csdl.sum(x0))
        y1 = y-csdl.exp(x1)**(x1*x1)
        y2 = csdl.tensordot(y1, x0, axes=([1],[1]),)
        with csdl.Namespace('deriv1'):
            dy_dx1 = csdl.derivative(y2, [x1, x0])[x1]
            dy_dx1.add_name('dy_dx1')

        y3 = csdl.maximum(csdl.absolute(dy_dx1/100000))
        compare_values = []
        # compare_values += [csdl_tests.TestingPair(dy_dx1, dy_dx1.value)]
        compare_values += [csdl_tests.TestingPair(y3, y3.value)]
        self.run_tests(compare_values=compare_values, verify_derivatives=True)

    def test_deriv_composed(self):
        """
        Test single derivatives with composed operations
        """
        self.prep()

        x0 = csdl.Variable(name = 'x0', value=1.0)
        x1 = csdl.Variable(name = 'x1', value=2.0)

        y2 = x1-x0
        y2.add_name('y2')
        with csdl.Namespace('deriv1'):
            dy_dx1 = csdl.derivative(y2, [x1, x0])[x1]
            dy_dx1.add_name('dy_dx1')

        with csdl.Namespace('deriv2'):
            dy_dx2 = csdl.derivative(dy_dx1, [x1])[x1]
            dy_dx2.add_name('dy2_dx2')

        compare_values = []
        compare_values += [csdl_tests.TestingPair(dy_dx2, dy_dx2.value)]
        compare_values += [csdl_tests.TestingPair(dy_dx1, np.ones((1,1)))]
        self.run_tests(compare_values=compare_values, verify_derivatives=True)

    def test_deriv_composed2(self):
        """
        Test single derivatives with composed operations
        """
        self.prep()

        x0 = csdl.Variable(name = 'x0', value=np.array([1.0, 2.0]))
        x1 = csdl.Variable(name = 'x1', value=np.array([3.0, -2.0]))

        # y2 = x1-x1
        y2 = csdl.tensordot(x1, x1)
        y2.add_name('y2')
        with csdl.Namespace('deriv1'):
            dy_dx1 = csdl.derivative(y2, [x1, x0])[x1]
            dy_dx1.add_name('dy_dx1')

        with csdl.Namespace('deriv2'):
            dy_dx2 = csdl.derivative(csdl.sum(dy_dx1), [x1])[x1]
            dy_dx2.add_name('dy2_dx2')

        compare_values = []
        compare_values += [csdl_tests.TestingPair(dy_dx2, dy_dx2.value)]
        compare_values += [csdl_tests.TestingPair(dy_dx1, dy_dx1.value)]
        self.run_tests(compare_values=compare_values, verify_derivatives=True)

if __name__ == '__main__':
    t = TestDeriv()
    t.test_deriv()
    t.test_deriv_2()
    t.test_deriv_3()
    # t.test_deriv_composed()
    t.test_deriv_composed2()






